"""
Gene alignment, target-side normalization, and signature scoring.

The teacher MLP is trained on bulk RNA-seq.  Applying the MLP directly to
single-cell expression is unreliable because the marginal distribution of
each gene is very different (bulk gene means ~ log2(TPM+1) of 5-10 vs sc
log1p(CPM) of 0-2 with 70-95% zeros).  Instead, we extract a signed
phenotype-program gene weight ``w_gene`` from the teacher (input * grad)
and compute target cell scores by

    s_pseudo(cell) = z(rank_score(target_X @ w_gene))

This is rank-based and therefore scale-invariant.  This score is the
distillation target for the student GNN.

Optional conditional VAE for batch correction is included but disabled by
default (only activated when batch_correction=True and batch_key is given).
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# gene alignment
# ---------------------------------------------------------------------------
def align_genes(
    bulk_expr: pd.DataFrame,
    adata,
    *,
    case_insensitive: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align bulk and target on shared genes.

    Returns
    -------
    bulk_X_aligned : (n_b, g_shared) float32 numpy
    target_idx     : indices into adata.var of shared genes (g_shared,)
    shared_genes   : list of gene names
    """
    log = logger or logging.getLogger("MAPPER")
    bulk_genes = list(bulk_expr.columns)
    target_genes = list(adata.var_names)

    if case_insensitive:
        b_map = {str(g).upper(): g for g in bulk_genes}
        t_map = {str(g).upper(): g for g in target_genes}
        shared_keys = sorted(set(b_map.keys()) & set(t_map.keys()))
        shared_bulk = [b_map[k] for k in shared_keys]
        shared_target = [t_map[k] for k in shared_keys]
    else:
        shared_set = sorted(set(bulk_genes) & set(target_genes))
        shared_bulk = shared_set
        shared_target = shared_set

    if len(shared_bulk) < 10:
        raise ValueError(
            f"only {len(shared_bulk)} shared genes between bulk and target — "
            f"check gene symbol convention"
        )
    if len(shared_bulk) < 200 and log is not None:
        log.warning(
            "only %d shared genes — teacher will be trained on a narrow panel "
            "(typical for proteomics/IMC targets). Scores remain valid but the "
            "phenotype program will reflect only the overlapping markers.",
            len(shared_bulk),
        )
    elif len(shared_bulk) < 1000 and log is not None:
        log.warning("only %d shared genes; consider using gene symbol harmonization",
                    len(shared_bulk))

    bulk_X = bulk_expr[shared_bulk].to_numpy(dtype=np.float32, copy=False)
    target_idx = np.array(
        [adata.var_names.get_loc(g) for g in shared_target], dtype=np.int64
    )
    log.info("[align] shared genes = %d", len(shared_bulk))
    return bulk_X, target_idx, shared_bulk


# ---------------------------------------------------------------------------
# bulk expression normalisation (column-wise z-score)
# ---------------------------------------------------------------------------
def zscore_columns(
    X: np.ndarray,
    clip: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score with optional clipping.

    Returns
    -------
    X_z   : (N, G) float32 z-scored matrix
    mean_ : (G,) column means
    std_  : (G,) column stds (floored at eps)
    """
    mean_ = X.mean(axis=0).astype(np.float32)
    std_  = X.std(axis=0).astype(np.float32)
    std_  = np.where(std_ < eps, 1.0, std_).astype(np.float32)
    X_z   = ((X - mean_) / std_).astype(np.float32)
    if clip > 0:
        X_z = np.clip(X_z, -clip, clip)
    return X_z, mean_, std_


# ---------------------------------------------------------------------------
# gene selection (Elastic Net / variance)
# ---------------------------------------------------------------------------
def select_genes(
    bulk_X: np.ndarray,
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    n_select: int = 2000,
    method: str = "elasticnet",
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    class_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    continuous_col: Optional[str] = None,
    random_seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Return sorted indices of the top ``n_select`` genes.

    Elastic Net fits a penalized model on bulk expression to find genes
    that are jointly discriminative for the phenotype.  Correlated gene
    groups are handled more stably than pure L1 (LASSO) because the L2
    term prevents arbitrary within-group dropping.

    Falls back to variance-based selection if the model fails.
    Returns ``np.arange(n_genes)`` unchanged when ``n_select >= n_genes``.
    """
    log = logger or logging.getLogger("MAPPER")
    n_genes = bulk_X.shape[1]

    if n_select <= 0 or n_select >= n_genes:
        return np.arange(n_genes, dtype=np.int64)

    log.info("[gene_select] selecting top %d / %d genes via %s", n_select, n_genes, method)

    if method == "variance":
        return _select_by_variance(bulk_X, n_select, log)

    y, task = _prepare_bulk_y(
        bulk_meta, phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        class_col=class_col,
        positive_class=positive_class,
        continuous_col=continuous_col,
    )
    try:
        coef = _fit_elasticnet(bulk_X, y, task, random_seed)
        importance = np.abs(coef).max(axis=0) if coef.ndim == 2 else np.abs(coef)
        n_nonzero = int((importance > 0).sum())
        sel = np.sort(np.argsort(importance)[::-1][:n_select]).astype(np.int64)
        log.info("[gene_select] elasticnet: %d non-zero coefs → kept top %d", n_nonzero, len(sel))
        return sel
    except Exception as e:
        log.warning("[gene_select] elasticnet failed (%s); falling back to variance.", e)
        return _select_by_variance(bulk_X, n_select, log)


def _select_by_variance(bulk_X: np.ndarray, n_select: int, log) -> np.ndarray:
    var = np.var(bulk_X, axis=0)
    sel = np.sort(np.argsort(var)[::-1][:n_select]).astype(np.int64)
    log.info("[gene_select] variance: kept top %d genes", len(sel))
    return sel


def _prepare_bulk_y(
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    class_col: Optional[str],
    positive_class: Optional[str],
    continuous_col: Optional[str],
):
    """Extract (y, task) from bulk_meta for gene selection model."""
    if phenotype_type == "survival":
        y = bulk_meta[survival_event_col].to_numpy(dtype=np.float32)
        return y, "binary"
    if phenotype_type in ("binary", "classification"):
        labels = bulk_meta[class_col].astype(str).to_numpy()
        if positive_class is not None:
            y = (labels == str(positive_class)).astype(np.float32)
        else:
            uniq = sorted(np.unique(labels).tolist())
            y = (labels == uniq[-1]).astype(np.float32)
        return y, "binary"
    if phenotype_type == "multiclass":
        labels = bulk_meta[class_col].astype(str).to_numpy()
        uniq = sorted(np.unique(labels).tolist())
        y = np.array([uniq.index(lbl) for lbl in labels], dtype=np.int64)
        return y, "multiclass"
    # continuous
    y = bulk_meta[continuous_col].to_numpy(dtype=np.float32)
    return y, "regression"


def _fit_elasticnet(
    bulk_X: np.ndarray, y: np.ndarray, task: str, random_seed: int
) -> np.ndarray:
    """Fit Elastic Net and return coefficient array shape (n_genes,) or (n_classes, n_genes)."""
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(bulk_X)

    if task == "regression":
        from sklearn.linear_model import ElasticNetCV
        m = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9], cv=5, max_iter=5000,
            random_state=random_seed, n_jobs=-1,
        )
        m.fit(Xs, y)
        return m.coef_   # (n_genes,)

    from sklearn.linear_model import LogisticRegressionCV
    m = LogisticRegressionCV(
        penalty="elasticnet",
        l1_ratios=[0.5, 0.9],
        solver="saga",
        cv=5,
        max_iter=2000,
        random_state=random_seed,
        n_jobs=-1,
        multi_class="ovr",
    )
    m.fit(Xs, y)
    return m.coef_   # (1, n_genes) binary or (n_classes, n_genes) multiclass


# ---------------------------------------------------------------------------
# target normalization
# ---------------------------------------------------------------------------
def normalize_target(
    adata,
    target_idx: np.ndarray,
    *,
    use_gpu: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """Return ``target_X``, a (N, G) row-normalized log1p(CPM) matrix.

    If GPU backends are available (rapids-singlecell + cupy) we use them;
    otherwise scanpy on CPU; otherwise a numpy fallback that preserves
    sparsity.

    The matrix is dense float32 on CPU for downstream torch transfer.
    For very large datasets the caller should chunk forward passes.
    """
    log = logger or logging.getLogger("MAPPER")
    X = adata.X
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    X_sub = X[:, target_idx]
    n_cells, n_genes = X_sub.shape

    if not _looks_like_counts(X_sub):
        out = X_sub.toarray() if sp.issparse(X_sub) else np.asarray(X_sub)
        out = np.asarray(out, dtype=np.float32)
        log.info("[normalize] input is already-normalized (non-count); "
                 "skipped CPM, returning float32 (%dx%d)", out.shape[0], out.shape[1])
        return out

    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cusp 
            import rapids_singlecell as rsc 
            return _normalize_target_gpu(X_sub, log)
        except Exception as e:
            log.warning("[transfer] GPU normalize failed (%s); falling back to CPU.", e)

    # CPU path -- robust to sparse / dense
    return _normalize_target_cpu(X_sub, log)


def _looks_like_counts(X) -> bool:
    """Heuristic: count data is non-negative and (almost) integer-valued."""
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return True
    if data.size > 100_000:
        rng = np.random.default_rng(0)
        data = data[rng.choice(data.size, 100_000, replace=False)]
    if np.any(data < 0):
        return False
    return bool(np.allclose(data, np.round(data), atol=1e-6))


def _normalize_target_cpu(X_sub, log) -> np.ndarray:
    if sp.issparse(X_sub):
        rs = np.asarray(X_sub.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        scale = (1e4 / rs).astype(np.float32)
        # multiply each row by its scale, then log1p
        X_norm = X_sub.multiply(scale[:, None]).tocsr()
        X_norm.data = np.log1p(X_norm.data, dtype=np.float32)
        out = np.asarray(X_norm.todense(), dtype=np.float32)
        log.info("[normalize] sparse log1p(CPM) -> dense (%dx%d)", out.shape[0], out.shape[1])
        return out
    X = np.asarray(X_sub, dtype=np.float32)
    rs = X.sum(axis=1)
    rs[rs == 0] = 1.0
    out = np.log1p(X * (1e4 / rs[:, None]).astype(np.float32)).astype(np.float32)
    log.info("[normalize] dense log1p(CPM) (%dx%d)", out.shape[0], out.shape[1])
    return out


def _normalize_target_gpu(X_sub, log) -> np.ndarray:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    if sp.issparse(X_sub):
        Xg = cusp.csr_matrix(X_sub.astype(np.float32))
    else:
        Xg = cp.asarray(X_sub, dtype=cp.float32)
    if cusp.issparse(Xg):
        rs = cp.asarray(Xg.sum(axis=1)).ravel()
        rs = cp.where(rs == 0, cp.float32(1.0), rs)
        scale = (cp.float32(1e4) / rs)
        Xg = Xg.multiply(scale[:, None])
        Xg = Xg.tocsr()
        Xg.data = cp.log1p(Xg.data)
        out = Xg.toarray()
    else:
        rs = Xg.sum(axis=1)
        rs = cp.where(rs == 0, cp.float32(1.0), rs)
        out = cp.log1p(Xg * (cp.float32(1e4) / rs[:, None]))
    out = cp.asnumpy(out).astype(np.float32)
    log.info("[normalize] GPU log1p(CPM) (%dx%d)", out.shape[0], out.shape[1])
    return out


# ---------------------------------------------------------------------------
# (optional) target-side conditional VAE -- exposed only via batch_correction=True
# ---------------------------------------------------------------------------
def maybe_correct_with_vae(
    target_X: np.ndarray,
    batch_labels: Optional[np.ndarray],
    *,
    enabled: bool = False,
    slide_ids: Optional[np.ndarray] = None,
    config=None,
    device=None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    if not enabled:
        return target_X
    if batch_labels is None:
        raise ValueError("batch_correction=True requires a batch_key")
    log = logger or logging.getLogger("MAPPER")
    log.info("[vae] running target-side conditional VAE for batch correction")
    from .vae import fit_conditional_vae   # lazy import; module is light
    return fit_conditional_vae(
        target_X, batch_labels, slide_ids=slide_ids,
        config=config, device=device, logger=log,
    )


# ---------------------------------------------------------------------------
# (optional) data-mode-specific reconstruction
#   scRNA  -> NB-decoder VAE on raw counts
#   stRNA  -> spatial GraphVAE on log1p(CPM) + spatial edges
# ---------------------------------------------------------------------------
def maybe_reconstruct(
    adata,
    target_X: np.ndarray,
    target_idx: np.ndarray,
    data_mode: str,
    *,
    enabled: bool = False,
    platform: str = "generic",
    spatial_radius_um: Optional[float] = None,
    config=None,
    device=None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Dispatch to data-mode-appropriate reconstruction VAE.

    Returns a denoised log1p(CPM)-style matrix of the same shape as ``target_X``.
    Returns the input unchanged if ``enabled`` is False or required inputs
    (e.g. spatial coords for stRNA) are missing.
    """
    if not enabled:
        return target_X
    log = logger or logging.getLogger("MAPPER")
    if config is None:
        raise ValueError("maybe_reconstruct requires a MapperConfig")

    if data_mode == "scRNA":
        # Need raw counts -- pull from adata.X (assumed counts before normalize).
        from .vae import fit_nb_vae
        X = adata.X
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        Xs = X[:, target_idx]
        if sp.issparse(Xs):
            Xs = Xs.toarray()
        counts = np.asarray(Xs, dtype=np.float32)
        # heuristic sanity check: integer-like values?
        if np.nanmax(counts) < 30 and not np.allclose(counts, np.round(counts), atol=1e-3):
            log.warning("[reconstruct] adata.X looks pre-normalised (max=%.2f, non-integer); "
                        "NB VAE expects raw counts. Reconstruction may be unreliable.",
                        float(np.nanmax(counts)))
        return fit_nb_vae(counts, config=config.nbvae, device=device, logger=log)

    # spatial / stRNA path
    from .vae import fit_graph_vae
    from .platform import resolve_default_radius_um, coords_to_micron
    from .graph import build_spatial_graph
    from .utils import detect_backends

    radius_um = resolve_default_radius_um(platform, spatial_radius_um)
    coords_um = coords_to_micron(adata, platform, logger=log)
    if coords_um is None or radius_um is None or radius_um <= 0:
        log.warning("[reconstruct] spatial coords/radius unavailable; skipping GraphVAE.")
        return target_X
    slide_ids = (
        adata.obs["slide_id"].to_numpy()
        if "slide_id" in adata.obs.columns else None
    )
    G_spatial = build_spatial_graph(
        coords_um, radius_um,
        max_neighbors=config.graph.max_neighbors,
        slide_ids=slide_ids,
        chunk=config.graph.spatial_chunk_size,
        use_torch_cluster=detect_backends().torch_cluster,
        device=device, logger=log,
    )
    if G_spatial is None:
        log.warning("[reconstruct] no valid spatial edges; skipping GraphVAE.")
        return target_X
    return fit_graph_vae(target_X, G_spatial, config=config.graphvae,
                         device=device, logger=log)
