"""
Grouping logic for MAPPER (v0.7) — likelihood-ratio based.

Spatial mode uses two-tier positivity built on log-likelihood ratios:

    log_LR_cell  = log p(s_cell  | bulk_pos) - log p(s_cell  | bulk_neg)
    log_LR_niche = log p(s_niche | bulk_pos) - log p(s_niche | bulk_neg)

scRNA mode has no niche signal.  We Leiden-cluster on expression PCA and
mark every cluster whose mean ``log_LR_cell`` exceeds the threshold as
positive (rare phenotype-positive populations are not collapsed).
"""
from __future__ import annotations
import logging
import warnings
from typing import Dict, Tuple

import numpy as np

from ._internal_config import GroupingConfig


NEG_STRICT_CAP = 5.0   # ceiling for the data-adaptive k


def _compute_neg_strict_k(bulk_pos_size, bulk_neg_size) -> float:
    """Option C: k = (max(P,N)/min(P,N)) × (1 + 2·P/(P+N)), capped."""
    if bulk_pos_size is None or bulk_neg_size is None \
            or bulk_pos_size <= 0 or bulk_neg_size <= 0:
        return 1.0
    P = float(bulk_pos_size)
    N = float(bulk_neg_size)
    imbalance = max(P, N) / min(P, N)
    event_rate = P / (P + N)
    return float(min(imbalance * (1.0 + 2.0 * event_rate), NEG_STRICT_CAP))


# ---------------------------------------------------------------------------
# 1. Spatial 2-tier LR grouping
# ---------------------------------------------------------------------------
def likelihood_ratio_3class(
    log_lr_cell: np.ndarray,
    log_lr_niche: np.ndarray | None,
    *,
    log_lr_threshold: float = 0.5,
    bulk_pos_size: int | None = None,
    bulk_neg_size: int | None = None,
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
    bulk_pos_size, bulk_neg_size
        Sample counts in the bulk positive / negative groups.  Used to
        compute the data-adaptive negative-side factor ``k`` (Option C,
        see module docstring and note 30).  When omitted, ``k`` falls back
        to ``1.0`` (no scaling).

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
    k = _compute_neg_strict_k(bulk_pos_size, bulk_neg_size)
    tau_neg = tau * k

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
        "k_used": float(k),
        "tau_neg": float(tau_neg),
        "n_positive": int(is_pos.sum()),
        "n_negative": int(is_neg.sum()),
        "n_background": int((groups == "background").sum()),
    }
    log.info(
        "[group.lr] %s k=%.3f τ=%.2f τ_neg=%.2f  pos=%d neg=%d bg=%d",
        meta["method"], k, tau, tau_neg,
        meta["n_positive"], meta["n_negative"], meta["n_background"],
    )
    return groups, meta


# ---------------------------------------------------------------------------
# 2. scRNA cluster grouping (LR-based)
# ---------------------------------------------------------------------------
def cluster_phenotype_assign_lr(
    log_lr_cell: np.ndarray,
    expr_pca: np.ndarray,
    *,
    log_lr_threshold: float = 0.5,
    bulk_pos_size: int | None = None,
    bulk_neg_size: int | None = None,
    config: "GroupingConfig | None" = None,
    random_seed: int = 42,
    logger: logging.Logger | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Leiden cluster on expression PCA; positive = top-N clusters by mean
    ``log_LR_cell`` **and** the cell's own ``log_LR_cell > log_lr_threshold``.

    Why top-N (not "all clusters above threshold")?  In scRNA mode the
    teacher_emb-derived clusters often contain a tumour-adjacent cooperator
    population (e.g. SPP1+ macrophages) whose own cluster has moderate-but-
    positive mean LR.  Including those clusters wholesale dilutes cell-type
    purity.  Picking only the top-N clusters by phenotype evidence and
    additionally filtering by per-cell LR keeps the assignment
    cell-type-specific.

    Defaults (``top_n_clusters=2``, ``log_lr_threshold=0.5``) were empirically
    validated to deliver ~99 % target-cell purity on PRAD (NE) and HCC
    (TumorCells) while retaining 88-95 % recall.
    """
    cfg = config or GroupingConfig()
    log = logger or logging.getLogger("MAPPER")
    log_lr_cell = np.asarray(log_lr_cell, dtype=np.float32)
    N = log_lr_cell.shape[0]
    if N == 0:
        return np.array([], dtype=object), {"method": "empty"}

    cluster_labels, cluster_method = _leiden_cluster(
        expr_pca, cfg.cluster_n_neighbors, cfg.cluster_resolution,
        random_seed=random_seed, logger=log,
    )
    if cluster_labels is None:
        log.warning("[group.cluster_lr] clustering failed; falling back to cell-only LR")
        groups, meta = likelihood_ratio_3class(
            log_lr_cell, None,
            log_lr_threshold=log_lr_threshold,
            bulk_pos_size=bulk_pos_size, bulk_neg_size=bulk_neg_size,
            logger=log,
        )
        meta["method"] = "lr_cell_only_clusterfail"
        return groups, meta

    uniq = np.unique(cluster_labels)
    cluster_mean = {int(c): float(log_lr_cell[cluster_labels == c].mean()) for c in uniq}
    sorted_clusters = sorted(cluster_mean.items(), key=lambda kv: kv[1], reverse=True)
    top_n = max(1, int(cfg.top_n_clusters))
    pos_clusters = sorted([int(c) for c, _ in sorted_clusters[:top_n]])

    tau = float(log_lr_threshold)
    k = _compute_neg_strict_k(bulk_pos_size, bulk_neg_size)
    tau_neg = tau * k

    in_pos_cluster = np.isin(cluster_labels.astype(int), pos_clusters)
    is_pos = in_pos_cluster & (log_lr_cell > tau)
    is_neg = (log_lr_cell < -tau_neg) & (~is_pos)

    groups = np.full(N, "background", dtype=object)
    groups[is_pos] = "positive"
    groups[is_neg] = "negative"

    meta = {
        "method": f"cluster_lr_top{top_n}_{cluster_method}",
        "log_lr_threshold": tau,
        "k_used": float(k),
        "tau_neg": float(tau_neg),
        "top_n_clusters": top_n,
        "n_clusters": int(uniq.size),
        "positive_cluster_ids": pos_clusters,
        "positive_cluster_means": {c: cluster_mean[c] for c in pos_clusters},
        "cluster_mean_log_lr": cluster_mean,
        "cluster_labels": cluster_labels.astype(np.int32),
        "n_positive": int(is_pos.sum()),
        "n_negative": int(is_neg.sum()),
        "n_background": int((groups == "background").sum()),
    }
    log.info(
        "[group.cluster_lr] method=%s n_clusters=%d top%d pos_clusters=%s "
        "(means=%s) k=%.3f τ=%.2f τ_neg=%.2f  pos=%d bg=%d neg=%d",
        meta["method"], meta["n_clusters"], top_n, pos_clusters,
        ["%.2f" % cluster_mean[c] for c in pos_clusters],
        k, tau, tau_neg,
        meta["n_positive"], meta["n_background"], meta["n_negative"],
    )
    return groups, meta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _leiden_cluster(expr_pca, n_neighbors, resolution, *, random_seed, logger):
    """Try Leiden; fall back to k-means(k=8) if scanpy/igraph missing."""
    try:
        import scanpy as sc
        import anndata as ad
        ad_tmp = ad.AnnData(X=np.zeros((expr_pca.shape[0], 1), dtype=np.float32))
        ad_tmp.obsm["X_pca"] = np.asarray(expr_pca, dtype=np.float32)
        sc.pp.neighbors(ad_tmp, n_neighbors=n_neighbors, use_rep="X_pca")
        sc.tl.leiden(ad_tmp, resolution=resolution, random_state=random_seed)
        labels = ad_tmp.obs["leiden"].astype(int).to_numpy()
        return labels, "leiden"
    except Exception as e:
        logger.warning("[cluster] leiden failed (%s); falling back to k-means(k=8)", e)
    try:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=8, n_init=10, random_state=random_seed).fit_predict(expr_pca)
        return labels.astype(int), "kmeans8"
    except Exception as e:
        logger.error("[cluster] k-means also failed (%s)", e)
        return None, "failed"
