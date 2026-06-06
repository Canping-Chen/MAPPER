"""
Niche-level phenotype scoring + bulk score distributions.

What this module provides
-------------------------
1. ``compute_local_pseudobulk``   — LOO neighbour-weighted pseudo-bulk.

2. ``compute_s_niche_direct``     — feed pseudo-bulk through the (frozen)
   teacher MLP to get ``s_niche``.  Same scale as ``s_cell``.

3. ``fit_bulk_score_distributions`` — run the teacher MLP on the bulk training
   matrix and split the resulting scores into phenotype-positive and
   phenotype-negative subsets.  These two empirical distributions are the
   reference for likelihood-ratio classification (replaces the old single-
   quantile ``bulk_*_thr`` design).

4. ``log_likelihood_ratio``       — for a vector of query scores, compute
   ``log p(score | bulk_pos) - log p(score | bulk_neg)`` via kernel density
   estimation.  ``log_LR > 0`` ⇒ evidence for phenotype+, ``< 0`` ⇒ evidence
   for phenotype-, ``≈ 0`` ⇒ ambiguous.

Why the LR formulation (instead of an absolute threshold on the MLP score)
-------------------------------------------------------------------------
A single quantile threshold (e.g. ``bulk_pos_thr = median(bulk_pos_scores)``)
ignores the fact that an entire bulk sample aggregates ~10^5–10^6 cells,
while a target spot/cell aggregates only ~1–10.  Comparing a per-spot score
against the aggregate-bulk median is a scale mismatch that systematically
rejects most spots even in obviously phenotype-positive tissue.

The likelihood ratio compares *distributions*, not points, so the scale
mismatch is absorbed naturally: a spot is "positive-like" when its score
profile is more consistent with the bulk-pos distribution than with bulk-neg,
regardless of absolute magnitude.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Local pseudo-bulk (LOO)
# ---------------------------------------------------------------------------
def compute_local_pseudobulk(
    target_X: np.ndarray,             # (N, G) float32
    edge_index: torch.Tensor,         # (2, E) src -> dst, G_spatial
    edge_weight: Optional[torch.Tensor] = None,
    include_self: bool = False,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Return ``x_pb`` (N, G) — neighbour-weighted local pseudo-bulk."""
    log = logger or logging.getLogger("MAPPER")
    N, G = target_X.shape
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    if edge_weight is None:
        w_e = np.ones(src.shape[0], dtype=np.float32)
    else:
        w_e = edge_weight.cpu().numpy().astype(np.float32)

    x_pb = np.zeros((N, G), dtype=np.float32)
    wsum = np.zeros(N, dtype=np.float32)
    chunk = max(1, int(2_000_000 // max(G, 1)))
    for s in range(0, src.shape[0], chunk):
        e = min(s + chunk, src.shape[0])
        np.add.at(x_pb, dst[s:e], target_X[src[s:e]] * w_e[s:e, None])
        np.add.at(wsum, dst[s:e], w_e[s:e])
    if include_self:
        x_pb += target_X
        wsum += 1.0
    no_nei = wsum <= 0
    if no_nei.any():
        x_pb[no_nei] = target_X[no_nei]
        wsum[no_nei] = 1.0
    x_pb /= wsum[:, None]
    log.info("[niche] local pseudo-bulk: N=%d G=%d edges=%d isolated=%d",
             N, G, src.shape[0], int(no_nei.sum()))
    return x_pb


# ---------------------------------------------------------------------------
# 2. s_niche via teacher MLP on pseudo-bulk
# ---------------------------------------------------------------------------
def compute_s_niche_direct(
    target_X: np.ndarray,
    edge_index: torch.Tensor,
    teacher_model: nn.Module,
    *,
    edge_weight: Optional[torch.Tensor] = None,
    include_self: bool = False,
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, dict]:
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = next(teacher_model.parameters()).device

    pseudo_bulk = compute_local_pseudobulk(
        target_X, edge_index, edge_weight,
        include_self=include_self, logger=log,
    )

    N = pseudo_bulk.shape[0]
    s_raw = np.empty(N, dtype=np.float32)
    teacher_model.eval()
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            xb = torch.from_numpy(np.ascontiguousarray(pseudo_bulk[s:e])).to(device)
            out, _ = teacher_model(xb)
            s_raw[s:e] = out.view(-1).detach().cpu().numpy()

    info = {
        "method": "teacher_MLP_on_LOO_pseudobulk",
        "include_self": bool(include_self),
        "s_niche_min": float(s_raw.min()),
        "s_niche_max": float(s_raw.max()),
        "s_niche_mean": float(s_raw.mean()),
        "s_niche_std": float(s_raw.std()),
    }
    log.info("[niche] s_niche_direct: N=%d range=[%.3f, %.3f] mean=%.3f std=%.3f",
             N, info["s_niche_min"], info["s_niche_max"],
             info["s_niche_mean"], info["s_niche_std"])
    return s_raw, info


# ---------------------------------------------------------------------------
# 3. Bulk score distributions (replaces compute_bulk_anchored_thresholds)
# ---------------------------------------------------------------------------
def fit_bulk_score_distributions(
    teacher_model: nn.Module,
    bulk_X_z: np.ndarray,
    bulk_meta,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    class_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    continuous_col: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(bulk_pos_scores, bulk_neg_scores, info)``.

    Both arrays are 1-D MLP outputs on the bulk training matrix, separated by
    phenotype label.  They serve as the empirical reference distributions for
    likelihood-ratio classification of target spots/cells.

    Subgroup definitions
        survival       : pos = events (event == 1), neg = censored
        classification : pos = positive_class (or rarer class if not given)
        continuous     : pos = top quartile, neg = bottom quartile
    """
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = next(teacher_model.parameters()).device

    teacher_model.eval()
    n_bulk = bulk_X_z.shape[0]
    bulk_scores = np.empty(n_bulk, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_bulk, batch_size):
            e = min(s + batch_size, n_bulk)
            xb = torch.from_numpy(np.ascontiguousarray(bulk_X_z[s:e])).to(device)
            out, _ = teacher_model(xb)
            bulk_scores[s:e] = out.view(-1).cpu().numpy()

    if phenotype_type == "survival":
        events = bulk_meta[survival_event_col].astype(float).to_numpy()
        bulk_pos = bulk_scores[events >= 0.5]
        bulk_neg = bulk_scores[events < 0.5]
    elif phenotype_type == "classification":
        labels = bulk_meta[class_col].astype(str).to_numpy()
        if positive_class is not None:
            pos_mask = labels == str(positive_class)
        else:
            uniq = sorted(np.unique(labels).tolist())
            counts = {c: int((labels == c).sum()) for c in uniq}
            pos_mask = labels == min(counts, key=counts.get)
        bulk_pos = bulk_scores[pos_mask]
        bulk_neg = bulk_scores[~pos_mask]
    elif phenotype_type == "continuous":
        values = bulk_meta[continuous_col].astype(float).to_numpy()
        q75 = float(np.quantile(values, 0.75))
        q25 = float(np.quantile(values, 0.25))
        bulk_pos = bulk_scores[values >= q75]
        bulk_neg = bulk_scores[values <= q25]
    else:
        raise ValueError(f"Unknown phenotype_type: {phenotype_type}")

    if bulk_pos.size < 5 or bulk_neg.size < 5:
        log.warning(
            "[bulk_dist] phenotype subgroups too small (n_pos=%d, n_neg=%d). "
            "Falling back to global IQR split.",
            bulk_pos.size, bulk_neg.size,
        )
        bulk_pos = bulk_scores[bulk_scores >= np.quantile(bulk_scores, 0.75)]
        bulk_neg = bulk_scores[bulk_scores <= np.quantile(bulk_scores, 0.25)]

    info = {
        "n_bulk": int(n_bulk),
        "n_bulk_pos": int(bulk_pos.size),
        "n_bulk_neg": int(bulk_neg.size),
        "bulk_pos_mean": float(bulk_pos.mean()),
        "bulk_pos_std": float(bulk_pos.std()),
        "bulk_pos_median": float(np.median(bulk_pos)),
        "bulk_neg_mean": float(bulk_neg.mean()),
        "bulk_neg_std": float(bulk_neg.std()),
        "bulk_neg_median": float(np.median(bulk_neg)),
    }
    log.info(
        "[bulk_dist] n_pos=%d (med=%.3f sd=%.3f) | n_neg=%d (med=%.3f sd=%.3f)",
        bulk_pos.size, info["bulk_pos_median"], info["bulk_pos_std"],
        bulk_neg.size, info["bulk_neg_median"], info["bulk_neg_std"],
    )
    return bulk_pos.astype(np.float32), bulk_neg.astype(np.float32), info


# ---------------------------------------------------------------------------
# 4. Likelihood-ratio scoring (KDE-based)
# ---------------------------------------------------------------------------
def log_likelihood_ratio(
    query_scores: np.ndarray,
    bulk_pos_scores: np.ndarray,
    bulk_neg_scores: np.ndarray,
    *,
    bandwidth: float | str = "scott",
    clip: float = 20.0,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Per-spot ``log p(score | bulk_pos) - log p(score | bulk_neg)``.

    KDE (Gaussian kernel, Scott's bandwidth by default) is used to estimate
    both densities.  Output is clipped to ``[-clip, clip]`` for numerical
    stability when the query falls into the far tails of either distribution.

    Interpretation
        log_LR > 0  : evidence for phenotype+ (LR > 1)
        log_LR < 0  : evidence for phenotype-
        |log_LR| < threshold  : ambiguous (background)
    """
    from scipy.stats import gaussian_kde
    log = logger or logging.getLogger("MAPPER")

    q = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    bp = np.asarray(bulk_pos_scores, dtype=np.float64).reshape(-1)
    bn = np.asarray(bulk_neg_scores, dtype=np.float64).reshape(-1)

    # KDE needs >1 unique value with non-zero variance; otherwise fall back to
    # a Gaussian fit using the sample mean/std.
    def _safe_kde_logpdf(samples, query):
        if samples.size < 2 or np.std(samples) < 1e-9:
            mu = float(samples.mean()) if samples.size else 0.0
            sd = max(float(samples.std()), 1e-3)
            return -0.5 * np.log(2 * np.pi * sd ** 2) \
                   - 0.5 * ((query - mu) / sd) ** 2
        kde = gaussian_kde(samples, bw_method=bandwidth)
        return kde.logpdf(query)

    log_p_pos = _safe_kde_logpdf(bp, q)
    log_p_neg = _safe_kde_logpdf(bn, q)
    log_lr = (log_p_pos - log_p_neg).astype(np.float32)
    log_lr = np.clip(log_lr, -clip, clip)

    log.info(
        "[log_LR] computed for N=%d queries. log_LR range=[%.3f, %.3f] mean=%.3f  "
        "frac(log_LR>0)=%.3f  frac(log_LR<0)=%.3f",
        q.size, float(log_lr.min()), float(log_lr.max()), float(log_lr.mean()),
        float((log_lr > 0).mean()), float((log_lr < 0).mean()),
    )
    return log_lr
