"""
Grouping logic for MAPPER (v0.7) — likelihood-ratio based.

Both modes use the same likelihood-ratio thresholding.  Spatial mode adds a
second (niche) tier built on log-likelihood ratios:

    log_LR_cell  = log p(s_cell  | bulk_pos) - log p(s_cell  | bulk_neg)
    log_LR_niche = log p(s_niche | bulk_pos) - log p(s_niche | bulk_neg)

scRNA mode has no niche signal, so it calls the same routine with
``log_lr_niche=None`` (cell-only thresholding, no clustering).

All modalities then pass through ``refine_groups_bimodal``: a 1-vs-2-component
GMM BIC test on the positive / negative subsets demotes a marginal sub-mode
(e.g. spillover just past the threshold) to ``background`` when the subset is
genuinely bimodal, and leaves unimodal subsets untouched.  It runs on the
*call-determining* evidence axis -- for positives the strongest tier
``max(log_LR_cell, log_LR_niche)`` (mirroring the OR rule), for negatives the
strongest tier ``min(log_LR_cell, log_LR_niche)`` (mirroring the AND rule), so
niche-driven calls are not mistaken for a marginal sub-mode.  In scRNA (no
niche) both axes reduce to ``log_LR_cell``.  The only modality difference left
is whether the niche tier exists at all -- dictated by the data, not by a
modality-specific grouping rule.
"""
from __future__ import annotations
import logging
from typing import Dict, Tuple

import numpy as np


FOI_MIN_BULK_POS = 20   # below this many bulk-positive samples the q-quantile
                        # is too noisy; fall back to the symmetric τ_neg = τ.


def _calibrate_tau_neg(
    bulk_pos_loglr: np.ndarray | None,
    tau: float,
    foi_q: float,
) -> Tuple[float, dict]:
    """False-omission-calibrated negative threshold.

    The negative region is ``{log_LR < -τ_neg}``.  We set τ_neg so that at most
    ``foi_q`` of the *bulk positive* reference falls into it::

        P(log_LR < -τ_neg | bulk_pos) ≤ foi_q
        ⇔  -τ_neg = quantile_q(log_LR | bulk_pos)
        ⇔  τ_neg  = -quantile_q(log_LR | bulk_pos)

    The more the bulk positive / negative score distributions overlap, the more
    low (or negative) log_LR values appear among the bulk positives, so the
    q-quantile drops and τ_neg deepens — monotone in distributional overlap.

    A floor of τ keeps the negative threshold at least as strict as the
    positive evidence level (so well-separated cohorts stay symmetric rather
    than letting negatives be called on weaker evidence than positives).  The
    upper bound is implicit in the ``log_lr_clip`` already applied upstream.

    Falls back to ``τ_neg = τ`` when fewer than ``FOI_MIN_BULK_POS`` bulk
    positives are available (quantile unstable).
    """
    meta = {"foi_q": float(foi_q), "n_bulk_pos": 0,
            "foi_realized": None, "floored": False, "fallback": False}
    if bulk_pos_loglr is None:
        meta["fallback"] = True
        return tau, meta
    lp = np.asarray(bulk_pos_loglr, dtype=np.float64).reshape(-1)
    meta["n_bulk_pos"] = int(lp.size)
    if lp.size < FOI_MIN_BULK_POS:
        meta["fallback"] = True
        return tau, meta
    q_lr = float(np.quantile(lp, foi_q))     # q-quantile of bulk-pos log_LR
    tau_neg_raw = -q_lr
    tau_neg = max(float(tau), tau_neg_raw)
    meta["floored"] = bool(tau_neg_raw < tau)
    # realized false-omission rate at the (possibly floored) threshold
    meta["foi_realized"] = float(np.mean(lp < -tau_neg))
    return tau_neg, meta


# ---------------------------------------------------------------------------
# 1. Spatial 2-tier LR grouping
# ---------------------------------------------------------------------------
def likelihood_ratio_3class(
    log_lr_cell: np.ndarray,
    log_lr_niche: np.ndarray | None,
    *,
    log_lr_threshold: float = 0.5,
    bulk_pos_loglr: np.ndarray | None = None,
    foi_q: float = 0.05,
    logger: logging.Logger | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Two-tier likelihood-ratio grouping.

    Parameters
    ----------
    log_lr_cell
        Per-spot ``log_LR`` computed from ``s_cell`` (intrinsic evidence).
    log_lr_niche
        Per-spot ``log_LR`` computed from ``s_niche`` (niche-membership
        evidence).  Pass ``None`` to skip the niche tier (e.g. scRNA mode).
    log_lr_threshold
        Positive-side evidence threshold τ in natural-log units (``0.5`` ≈
        Bayes factor 1.65, moderate evidence).
    bulk_pos_loglr
        ``log_LR`` evaluated at the *bulk positive* reference scores (same KDE
        pair as ``log_lr_cell``).  Used to calibrate the negative-side
        threshold τ_neg to a false-omission rate ``foi_q`` (see
        ``_calibrate_tau_neg``).  When ``None`` / too few samples, τ_neg falls
        back to the symmetric value τ.
    foi_q
        Target false-omission rate for negative calls: at most ``foi_q`` of the
        bulk positives may fall into the negative region ``{log_LR < -τ_neg}``.

    Returns
    -------
    groups : (N,) ``{"negative", "background", "positive"}``
    meta   : dict with counts and parameters
    """
    log = logger or logging.getLogger("MAPPER")
    log_lr_cell = np.asarray(log_lr_cell, dtype=np.float32)
    N = log_lr_cell.shape[0]
    if N == 0:
        return np.array([], dtype=object), {"method": "empty"}

    if log_lr_niche is None:
        niche_used = False
    else:
        log_lr_niche = np.asarray(log_lr_niche, dtype=np.float32)
        niche_used = True

    tau = float(log_lr_threshold)
    tau_neg, foi_meta = _calibrate_tau_neg(bulk_pos_loglr, tau, foi_q)

    cell_pos  = log_lr_cell  > +tau
    niche_pos = (log_lr_niche > +tau) if niche_used else np.zeros(N, dtype=bool)
    is_pos = cell_pos | niche_pos

    cell_neg  = log_lr_cell < -tau_neg
    niche_neg = (log_lr_niche < -tau_neg) if niche_used else np.ones(N, dtype=bool)
    is_neg = (cell_neg & niche_neg) & (~is_pos)

    groups = np.full(N, "background", dtype=object)
    groups[is_pos] = "positive"
    groups[is_neg] = "negative"

    meta = {
        "method": "lr_2tier" if niche_used else "lr_cell_only",
        "log_lr_threshold": tau,
        "tau_neg": float(tau_neg),
        "n_positive": int(is_pos.sum()),
        "n_negative": int(is_neg.sum()),
        "n_background": int((groups == "background").sum()),
    }
    meta.update(foi_meta)
    log.info(
        "[group.lr] %s τ=%.2f τ_neg=%.2f (foi_q=%.3f realized=%s floored=%s n_bulk_pos=%d%s)  "
        "pos=%d neg=%d bg=%d",
        meta["method"], tau, tau_neg, foi_meta["foi_q"],
        ("%.3f" % foi_meta["foi_realized"]) if foi_meta["foi_realized"] is not None else "NA",
        foi_meta["floored"], foi_meta["n_bulk_pos"],
        " FALLBACK" if foi_meta["fallback"] else "",
        meta["n_positive"], meta["n_negative"], meta["n_background"],
    )
    return groups, meta


# ---------------------------------------------------------------------------
# 2. Modality-agnostic GMM/BIC refinement of positive & negative calls
# ---------------------------------------------------------------------------
def refine_groups_bimodal(
    groups: np.ndarray,
    pos_score: np.ndarray,
    neg_score: np.ndarray,
    *,
    pos_protect: np.ndarray | None = None,
    neg_protect: np.ndarray | None = None,
    min_cells: int = 20,
    bic_margin: float = 0.0,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Tighten positive / negative calls with a 1-vs-2-component GMM BIC test
    on the *call-determining* evidence axis (same rule for every modality).

    ``pos_score`` / ``neg_score`` are the per-cell evidence axes that decide the
    positive / negative calls respectively.  For spatial these are
    ``max(log_LR_cell, log_LR_niche)`` (mirroring the OR rule for positives) and
    ``min(log_LR_cell, log_LR_niche)`` (mirroring the AND rule for negatives),
    so a niche-driven call sits in the dominant component and is not mistaken
    for a marginal sub-mode.  For scRNA (no niche) both reduce to
    ``log_LR_cell``.

    ``pos_protect`` / ``neg_protect`` are optional boolean masks (length N,
    ``True`` = never demote this cell).  The refinement only removes
    *uncorroborated* marginal sub-modes: in spatial, a marginal call that the
    *other* tier independently supports is a real boundary/transition cell, not
    spillover, so it is protected (``pos_protect = niche_pos``,
    ``neg_protect = niche_neg``; the latter is True for every spatial negative
    by the AND rule, so spatial negatives are never demoted).  In scRNA no
    corroboration is available, so the masks are ``None`` and the test cleans
    both subsets as before.

    For the positive and negative subsets *independently*:

      * fit ``GaussianMixture(1)`` and ``GaussianMixture(2)`` on the subset's
        scores and compare their BIC (lower is better);
      * if ``bic2 + bic_margin < bic1`` the subset is genuinely bimodal — keep
        only the dominant-evidence component (the higher-score component for
        ``positive``, the lower-score component for ``negative``) and demote
        the other component's cells to ``"background"``;
      * otherwise the subset is unimodal and is left unchanged.

    Subsets with fewer than ``min_cells`` cells (or degenerate, single-valued
    scores) are skipped — a 2-component GMM is unstable there.  Any sklearn
    import / fit failure falls back to no refinement (groups returned as-is).

    Returns ``(groups, meta)``; ``groups`` is a refined copy.
    """
    log = logger or logging.getLogger("MAPPER")
    groups = np.array(groups, dtype=object)          # work on a copy
    pos_score = np.asarray(pos_score, dtype=np.float64).reshape(-1)
    neg_score = np.asarray(neg_score, dtype=np.float64).reshape(-1)
    meta: Dict = {
        "method": "bimodal_bic",
        "min_cells": int(min_cells),
        "bic_margin": float(bic_margin),
        "positive": None,
        "negative": None,
    }

    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        log.warning("[group.gmm] sklearn unavailable (%s); skipping refinement", e)
        meta["method"] = "bimodal_bic_skipped_no_sklearn"
        return groups, meta

    def _refine_one(label: str, score: np.ndarray, keep_high: bool,
                    protect: np.ndarray | None) -> Dict:
        idx = np.where(groups == label)[0]
        sub: Dict = {
            "n_in": int(idx.size), "refined": False,
            "bic1": None, "bic2": None, "n_demoted": 0, "n_protected": 0,
        }
        if idx.size < min_cells:
            sub["reason"] = "too_few_cells"
            return sub
        s = score[idx].reshape(-1, 1)
        if np.unique(s).size < 2:
            sub["reason"] = "degenerate_scores"
            return sub
        try:
            g1 = GaussianMixture(n_components=1, random_state=random_seed).fit(s)
            g2 = GaussianMixture(n_components=2, n_init=5, random_state=random_seed).fit(s)
            bic1 = float(g1.bic(s)); bic2 = float(g2.bic(s))
        except Exception as e:
            sub["reason"] = "fit_failed:%s" % e
            return sub
        sub["bic1"] = bic1
        sub["bic2"] = bic2
        if not (bic2 + bic_margin < bic1):
            sub["reason"] = "unimodal"
            return sub
        # bimodal -> keep the dominant-evidence component, demote the other
        comp = g2.predict(s)
        means = g2.means_.reshape(-1)
        keep_comp = int(np.argmax(means)) if keep_high else int(np.argmin(means))
        demote = idx[comp != keep_comp]
        # spare cells the other tier independently corroborates (real boundary,
        # not spillover): protected marginal cells stay in their class
        if protect is not None and demote.size:
            keep_mask = protect[demote]
            sub["n_protected"] = int(keep_mask.sum())
            demote = demote[~keep_mask]
        groups[demote] = "background"
        sub["refined"] = True
        sub["n_demoted"] = int(demote.size)
        sub["kept_mean"] = float(means[keep_comp])
        sub["demoted_mean"] = float(means[1 - keep_comp])
        return sub

    meta["positive"] = _refine_one("positive", pos_score, keep_high=True,
                                   protect=pos_protect)
    meta["negative"] = _refine_one("negative", neg_score, keep_high=False,
                                   protect=neg_protect)

    log.info(
        "[group.gmm] bimodal refine: pos(refined=%s demoted=%d protected=%d) "
        "neg(refined=%s demoted=%d protected=%d)",
        meta["positive"]["refined"], meta["positive"]["n_demoted"], meta["positive"]["n_protected"],
        meta["negative"]["refined"], meta["negative"]["n_demoted"], meta["negative"]["n_protected"],
    )
    return groups, meta
