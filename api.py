"""
MAPPER public API (v0.7).

Methodology
-----------
The bulk teacher defines the intrinsic phenotype program.  The physical
spatial graph defines where the signal can be spatially refined.  The
niche is scored by feeding the local pseudo-bulk through the same teacher
(non-linear, bulk-comparable scale).  Grouping uses likelihood-ratio
classification against the bulk score distributions, with two-tier
positivity (cell-intrinsic OR niche-membership) and a data-adaptive
negative threshold.

Spatial / stRNA mode
    G_spatial    = physical KNN-radius graph in microns (no fusion)
    s_cell       = teacher_MLP(target_X)
    s_niche      = teacher_MLP( mean(target_X[j], j ∈ N(i)) )
    log_LR_cell  = log p(s_cell  | bulk_pos) - log p(s_cell  | bulk_neg)
    log_LR_niche = log p(s_niche | bulk_pos) - log p(s_niche | bulk_neg)

    τ_neg = τ × k,  k = min( (max(P,N)/min(P,N)) × (1 + 2·P/(P+N)), 5 )
    where P, N are bulk pos / neg sample counts (Option C, see note 30).

    pos = (log_LR_cell  > +τ)    OR  (log_LR_niche > +τ)
    neg = (log_LR_cell  < -τ_neg) AND (log_LR_niche < -τ_neg)  AND not pos
    bg  = otherwise

scRNA mode
    No physical graph.  Leiden cluster on expression PCA, then mark every
    cluster whose mean ``log_LR_cell`` exceeds the threshold as positive.

Outputs
-------
Two columns are written to ``adata.obs`` — that is all a typical user needs::

    adata.obs["MAPPER.Score"]   float       composite, raw bulk-comparable scale
    adata.obs["MAPPER.Group"]   Categorical {negative, background, positive}

All intermediate quantities live in ``adata.uns["MAPPER_meta"]`` for
reproducibility / advanced analysis (LR-based filtering, ablations, etc.)::

    adata.uns["MAPPER_meta"]["per_cell"] = {
        "s_cell":   (N,) float   intrinsic teacher score
        "s_niche":  (N,) float   pseudo-bulk niche score      (spatial only)
        "LR_cell":  (N,) float   log Bayes factor on s_cell
        "LR_niche": (N,) float   log Bayes factor on s_niche  (spatial only)
    }
"""
from __future__ import annotations
import copy
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from . import _internal_config as _cfg
from .utils import (
    auto_device, get_logger, set_seed, detect_backends,
    gpu_available, list_cuda_devices, free_gpu_memory,
    pick_best_cuda_device,
)
from .platform import (
    is_spatial_platform,
    resolve_default_radius_um,
    coords_to_micron,
)
from .teacher import fit_teacher
from .transfer import (
    align_genes,
    select_genes,
    zscore_columns,
    normalize_target,
    maybe_correct_with_vae,
    maybe_reconstruct,
)
from .graph import build_spatial_graph
from .niche import (
    compute_s_niche_direct,
    fit_bulk_score_distributions,
    log_likelihood_ratio,
)
from .student import fit_student_distill
from .grouping import (
    likelihood_ratio_3class,
    cluster_phenotype_assign_lr,
)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def mapper(
    adata,
    bulk_expr: pd.DataFrame,
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    class_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    continuous_col: Optional[str] = None,
    data_mode: str = "spatial",
    platform: str = "visium",
    spatial_radius_um: Optional[float] = None,
    random_seed: int = 42,
    verbose: bool = True,
    batch_correction: bool = False,
    batch_key: Optional[str] = None,
    _config: Optional[_cfg.MapperConfig] = None,
):
    """Run MAPPER on a single AnnData."""
    cfg = _config or _cfg.DEFAULT
    log = get_logger(verbose)
    set_seed(random_seed)

    if data_mode not in {"spatial", "scRNA"}:
        raise ValueError(f"data_mode must be 'spatial' or 'scRNA', got {data_mode!r}")
    if platform not in {"visium", "visium_hd", "xenium", "generic", "scRNA"}:
        raise ValueError(f"unsupported platform: {platform!r}")

    log.info("=" * 60)
    log.info("MAPPER v0.7  data_mode=%s  platform=%s  N=%d  refine=%s α=%.2f τ=%.2f",
             data_mode, platform, adata.n_obs,
             cfg.niche.refine_method, cfg.niche.refine_alpha,
             cfg.niche.log_lr_threshold)
    log.info("=" * 60)

    # ---- 1. gene alignment + normalisation ----
    bulk_X, target_idx, shared_genes = align_genes(bulk_expr, adata, logger=log)

    if cfg.n_select_genes > 0:
        sel_idx = select_genes(
            bulk_X, bulk_meta, phenotype_type,
            n_select=cfg.n_select_genes,
            method=cfg.gene_select_method,
            survival_time_col=survival_time_col,
            survival_event_col=survival_event_col,
            class_col=class_col,
            positive_class=positive_class,
            continuous_col=continuous_col,
            random_seed=random_seed, logger=log,
        )
        bulk_X = bulk_X[:, sel_idx]
        target_idx = target_idx[sel_idx]
        shared_genes = [shared_genes[i] for i in sel_idx]

    bulk_X_z, _, _ = zscore_columns(bulk_X, clip=10.0)
    log.info("[normalize] bulk z-scored (%dx%d)", bulk_X_z.shape[0], bulk_X_z.shape[1])

    use_gpu = gpu_available() and adata.n_obs >= cfg.graph.gpu_n_threshold
    target_X = normalize_target(adata, target_idx, use_gpu=use_gpu, logger=log)

    if cfg.reconstruct:
        target_X = maybe_reconstruct(
            adata, target_X, target_idx, data_mode,
            enabled=True, platform=platform,
            spatial_radius_um=spatial_radius_um,
            config=cfg, device=auto_device(), logger=log,
        )

    target_X_z, _, _ = zscore_columns(target_X, clip=10.0)

    if batch_correction:
        if batch_key is None:
            raise ValueError("batch_correction=True requires batch_key")
        if batch_key not in adata.obs.columns:
            raise ValueError(f"batch_key={batch_key!r} not in adata.obs")
        batch_labels = adata.obs[batch_key].to_numpy()
        target_X = maybe_correct_with_vae(
            target_X, batch_labels,
            enabled=True, config=cfg.vae,
            device=auto_device(), logger=log,
        )
        target_X_z, _, _ = zscore_columns(target_X, clip=10.0)

    # ---- 2. Teacher (bulk training) ----
    device = auto_device()
    teacher_res = fit_teacher(
        bulk_X_z, bulk_meta, phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        class_col=class_col,
        positive_class=positive_class,
        continuous_col=continuous_col,
        config=cfg.teacher, device=device,
        random_seed=random_seed, logger=log,
    )

    try:
        del target_X
    except NameError:
        pass
    gc.collect()

    # ---- 4. Bulk score distributions (KDE reference) ----
    try:
        bulk_pos_scores, bulk_neg_scores, bulk_dist_info = fit_bulk_score_distributions(
            teacher_res.model, bulk_X_z, bulk_meta, phenotype_type,
            survival_time_col=survival_time_col,
            survival_event_col=survival_event_col,
            class_col=class_col, positive_class=positive_class,
            continuous_col=continuous_col,
            device=device, logger=log,
        )
    except Exception as e:
        log.error("[mapper] cannot fit bulk score distributions (%s)", e)
        bulk_pos_scores = bulk_neg_scores = None
        bulk_dist_info = {"error": str(e)}

    # ---- 5-9. Score + group this slide ----
    out = _score_and_group_one_slide(
        adata=adata, target_X_z=target_X_z,
        teacher_res=teacher_res,
        bulk_pos_scores=bulk_pos_scores, bulk_neg_scores=bulk_neg_scores,
        cfg=cfg, data_mode=data_mode, platform=platform,
        spatial_radius_um=spatial_radius_um,
        device=device, random_seed=random_seed, logger=log,
    )

    # release target_X_z
    try:
        del target_X_z
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 9. Write top-level uns metadata ----
    adata.uns["MAPPER_meta"] = {
        "version": "0.7",
        "n_obs": int(adata.n_obs),
        "n_genes_shared": int(len(shared_genes)),
        "data_mode": data_mode,
        "platform": platform,
        "spatial_used": bool(out["ran_spatial"]),
        "refinement": out["refinement"],
        "refine_method": cfg.niche.refine_method,
        "refine_alpha": out["refine_alpha"],
        "log_lr_threshold": float(cfg.niche.log_lr_threshold),
        "bulk_distribution_info": bulk_dist_info,
        "teacher_metric": teacher_res.metric_name,
        "teacher_val_metric": float(teacher_res.val_metric),
        "phenotype_type": phenotype_type,
        "group_meta": out["group_meta"],
        "per_cell": out["per_cell"],
    }
    groups = out["groups"]
    log.info("[mapper] DONE.  groups=%s",
             {k: int((groups == k).sum()) for k in ["negative", "background", "positive"]})
    return adata


# ---------------------------------------------------------------------------
# multi-slide entry
# ---------------------------------------------------------------------------
def mapper_multi_slide(
    adatas: Sequence,
    bulk_expr: pd.DataFrame,
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    class_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    continuous_col: Optional[str] = None,
    platform: str = "visium",
    spatial_radius_um: Optional[float] = None,
    random_seed: int = 42,
    verbose: bool = True,
    return_concat: bool = True,
    batch_correction: bool = False,
    batch_key: Optional[str] = None,
    _config: Optional[_cfg.MapperConfig] = None,
):
    """Process multiple spatial slides with **shared teacher + cross-slide
    normalisation**.

    Phenotype reference is shared, expression handling is configurable
    --------------------------------------------------------------------
    The teacher MLP and bulk KDE distributions are trained **once** so every
    slide is decoded against the same phenotype reference; this is what
    makes per-slide scores cross-slide comparable.

    Target expression handling is controlled by two orthogonal knobs:

      * ``cfg.multi_slide_normalization`` (default ``'joint'``):
          - ``'joint'``  : cross-slide z-score with shared mean/std
            (preserves cross-slide content differences — required for
            PAAD where Pt-12A (mostly normal) should have ~0 % positive
            and dense tumour slides should have many).
          - ``'per_slide'`` : each slide z-scores itself (within-slide
            relative ranking only; cross-slide content differences are
            erased).

      * ``cfg.reconstruct`` (default ``False``):
            per-slide GraphVAE denoising of low-quality slides.  Cross-slide
            edges are blocked inside ``build_spatial_graph`` via ``slide_id``,
            so each slide is denoised within its own spatial topology.
            **Restriction**: ``reconstruct=True`` is incompatible with
            ``multi_slide_normalization='joint'`` in v1 (joint stats would
            need a second pass over reconstructed values); the function
            silently falls back to ``'per_slide'`` with a warning.

    Batch correction
    ----------------
    ``batch_correction=True`` + ``batch_key='<obs column>'`` enables a global
    conditional VAE over all slides concatenated.  The VAE handles both
    cross-slide batches (each slide a single batch) and within-slide batches
    (multiple batches per slide) uniformly — only ``batch_labels`` matter.

    Memory budget: if estimated concat size (``N_total × G × 4 B``) exceeds
    10 GB, falls back to per-slide conditional VAE training (less effective
    at removing cross-slide batch effects but avoids OOM).

    Safety guard: if ``batch_key`` values are 1:1 with ``slide_id``,
    ``fit_conditional_vae`` emits a warning (it would erase all cross-slide
    variation including real biology) but does NOT block — the user is
    responsible for ensuring slides are technical replicates in that case.
    """
    import anndata as ad
    log = get_logger(verbose)
    cfg = _config or _cfg.DEFAULT
    set_seed(random_seed)
    log.info("=" * 60)
    log.info("MAPPER v0.7 multi-slide  N_slides=%d  platform=%s",
             len(adatas), platform)
    log.info("=" * 60)

    # ---- 1. Per-slide preparation: slide_id + micron coords ----
    used_ids: List[str] = []
    prepared = []
    for i, a in enumerate(adatas):
        a2 = a.copy()
        if "slide_id" not in a2.obs.columns:
            sid = f"slide_{i}"
        else:
            sid = str(a2.obs["slide_id"].iloc[0])
        base = sid; k = 1
        while sid in used_ids:
            sid = f"{base}_{k}"; k += 1
        used_ids.append(sid)
        a2.obs["slide_id"] = sid
        if platform in {"visium", "visium_hd"}:
            coords_um = coords_to_micron(a2, platform, logger=log)
            if coords_um is not None:
                a2.obsm["spatial"] = coords_um.astype(np.float32)
        prepared.append(a2)
    log.info("[multi] slide_ids=%s", used_ids)

    # ---- 2. Gene alignment: use bulk ∩ (∩ all slide var_names) ----
    common_target_genes = set(prepared[0].var_names)
    for a in prepared[1:]:
        common_target_genes &= set(a.var_names)
    bulk_genes = set(bulk_expr.columns)
    shared_genes = sorted(common_target_genes & bulk_genes)
    if len(shared_genes) < 100:
        raise ValueError(f"only {len(shared_genes)} shared genes across all "
                         "slides + bulk; cannot run multi-slide.")
    log.info("[multi] gene alignment: %d slides + bulk → %d shared genes",
             len(prepared), len(shared_genes))

    bulk_X = bulk_expr[shared_genes].values.astype(np.float32)
    if cfg.n_select_genes > 0:
        sel_idx = select_genes(
            bulk_X, bulk_meta, phenotype_type,
            n_select=cfg.n_select_genes,
            method=cfg.gene_select_method,
            survival_time_col=survival_time_col,
            survival_event_col=survival_event_col,
            class_col=class_col,
            positive_class=positive_class,
            continuous_col=continuous_col,
            random_seed=random_seed, logger=log,
        )
        bulk_X = bulk_X[:, sel_idx]
        shared_genes = [shared_genes[i] for i in sel_idx]
        log.info("[multi] gene pre-selection: %d → %d", bulk_X.shape[1], len(shared_genes))

    bulk_X_z, bulk_mean, bulk_std = zscore_columns(bulk_X, clip=10.0)
    log.info("[multi] bulk z-scored (%dx%d)", bulk_X_z.shape[0], bulk_X_z.shape[1])

    norm_mode = str(cfg.multi_slide_normalization).lower()
    if norm_mode not in {"per_slide", "joint"}:
        raise ValueError(f"multi_slide_normalization must be one of "
                         "{joint, per_slide}; got %r" % norm_mode)
    # reconstruct + joint is unsupported in v1 (joint stats would need a
    # second pass after reconstruction); fall back to per_slide with a warn.
    if cfg.reconstruct and norm_mode == "joint":
        log.warning("[multi] reconstruct=True + multi_slide_normalization='joint' "
                    "is not supported in v1; falling back to 'per_slide'.")
        norm_mode = "per_slide"
    log.info("[multi] target normalisation mode: %s", norm_mode)

    # batch_correction validation
    if batch_correction:
        if batch_key is None:
            raise ValueError("batch_correction=True requires batch_key")
        for sid, a in zip(used_ids, prepared):
            if batch_key not in a.obs.columns:
                raise ValueError(f"batch_key={batch_key!r} missing in slide {sid!r}")
        log.info("[multi] batch_correction enabled (key=%r)", batch_key)

    # ---- 3. Train teacher ONCE ----
    device = auto_device()
    teacher_res = fit_teacher(
        bulk_X_z, bulk_meta, phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        class_col=class_col,
        positive_class=positive_class,
        continuous_col=continuous_col,
        config=cfg.teacher, device=device,
        random_seed=random_seed, logger=log,
    )
    log.info("[multi] shared teacher trained (val %s=%.4f)",
             teacher_res.metric_name, teacher_res.val_metric)

    # ---- 4. Compute bulk KDE ONCE ----
    try:
        bulk_pos_scores, bulk_neg_scores, bulk_dist_info = fit_bulk_score_distributions(
            teacher_res.model, bulk_X_z, bulk_meta, phenotype_type,
            survival_time_col=survival_time_col,
            survival_event_col=survival_event_col,
            class_col=class_col, positive_class=positive_class,
            continuous_col=continuous_col,
            device=device, logger=log,
        )
    except Exception as e:
        log.error("[multi] cannot fit bulk score distributions (%s)", e)
        bulk_pos_scores = bulk_neg_scores = None
        bulk_dist_info = {"error": str(e)}

    # ---- 5. Process each slide ----
    tau = float(cfg.niche.log_lr_threshold)
    refine_method = cfg.niche.refine_method

    # ---- 5a. batch_correction pre-pass (if enabled) ----
    # When batch_correction is on, we must collect all per-slide target_X
    # (post-normalize, post-reconstruct) in RAM so the conditional VAE sees
    # the full cross-slide distribution.  Cache is then consumed by the
    # joint-stats pass (5b) and the main per-slide loop (5c).
    target_X_cache = {}   # sid -> np.ndarray  (only populated if batch_correction)
    MAX_CONCAT_GB = 10.0  # memory budget for the global VAE concat
    if batch_correction:
        log.info("[multi.vae] pre-collecting per-slide target_X for global VAE")
        for sid, a in zip(used_ids, prepared):
            var_pos_a = pd.Series(np.arange(a.n_vars), index=a.var_names)
            idx_a = var_pos_a.loc[shared_genes].values.astype(np.int64)
            use_gpu_a = gpu_available() and a.n_obs >= cfg.graph.gpu_n_threshold
            tx = normalize_target(a, idx_a, use_gpu=use_gpu_a, logger=log)
            if cfg.reconstruct:
                tx = maybe_reconstruct(
                    adata=a, target_X=tx, target_idx=idx_a,
                    data_mode="spatial", enabled=True, platform=platform,
                    spatial_radius_um=spatial_radius_um,
                    config=cfg, device=device, logger=log,
                )
            target_X_cache[sid] = tx

        G_full = next(iter(target_X_cache.values())).shape[1]
        N_total = sum(v.shape[0] for v in target_X_cache.values())
        est_gb = N_total * G_full * 4 / 1e9
        log.info("[multi.vae] global concat: N_total=%d G=%d est=%.2f GB",
                 N_total, G_full, est_gb)

        if est_gb > MAX_CONCAT_GB:
            log.warning("[multi.vae] concat size %.2f GB > %.1f GB budget; "
                        "falling back to per-slide batch correction (less "
                        "effective at removing cross-slide batch effects).",
                        est_gb, MAX_CONCAT_GB)
            for sid, a in zip(used_ids, prepared):
                bl = a.obs[batch_key].to_numpy()
                target_X_cache[sid] = maybe_correct_with_vae(
                    target_X_cache[sid], bl, enabled=True,
                    slide_ids=None,   # within-slide → alignment check meaningless
                    config=cfg.vae, device=device, logger=log,
                )
        else:
            X_concat = np.concatenate(
                [target_X_cache[s] for s in used_ids], axis=0,
            )
            bl_concat = np.concatenate(
                [prepared[i].obs[batch_key].to_numpy()
                 for i in range(len(used_ids))], axis=0,
            )
            sid_concat = np.concatenate(
                [np.full(target_X_cache[s].shape[0], s, dtype=object)
                 for s in used_ids], axis=0,
            )
            X_corr = maybe_correct_with_vae(
                X_concat, bl_concat, enabled=True,
                slide_ids=sid_concat,
                config=cfg.vae, device=device, logger=log,
            )
            offset = 0
            for s in used_ids:
                n = target_X_cache[s].shape[0]
                target_X_cache[s] = X_corr[offset:offset + n]
                offset += n
            del X_concat, X_corr, bl_concat, sid_concat
            gc.collect()

    # ---- 5b. joint mode: cross-slide mean/std ----
    joint_mean = joint_std = None
    if norm_mode == "joint":
        if target_X_cache:
            # batch_correction populated the cache; compute stats from corrected
            log.info("[multi] joint-mode: stats from batch-corrected matrices")
            all_X = np.concatenate(
                [target_X_cache[s] for s in used_ids], axis=0,
            ).astype(np.float64)
            joint_mean = all_X.mean(axis=0).astype(np.float32)
            joint_std  = all_X.std(axis=0).astype(np.float32)
            del all_X
            gc.collect()
        else:
            # streaming pass (O(G) memory) on raw normalised matrices
            log.info("[multi] joint-mode: streaming cross-slide mean/std (one pass)")
            G = len(shared_genes)
            joint_n = 0
            joint_sum   = np.zeros(G, dtype=np.float64)
            joint_sumsq = np.zeros(G, dtype=np.float64)
            for sid, a in zip(used_ids, prepared):
                var_pos_a = pd.Series(np.arange(a.n_vars), index=a.var_names)
                idx_a = var_pos_a.loc[shared_genes].values.astype(np.int64)
                use_gpu_a = gpu_available() and a.n_obs >= cfg.graph.gpu_n_threshold
                tx = normalize_target(a, idx_a, use_gpu=use_gpu_a, logger=log)
                txd = tx.astype(np.float64)
                joint_n     += tx.shape[0]
                joint_sum   += txd.sum(axis=0)
                joint_sumsq += (txd * txd).sum(axis=0)
                del tx, txd
                gc.collect()
            joint_mean = (joint_sum / joint_n).astype(np.float32)
            joint_var  = joint_sumsq / joint_n - (joint_sum / joint_n) ** 2
            joint_std  = np.sqrt(np.maximum(joint_var, 0.0)).astype(np.float32)
        log.info("[multi.joint] mean range=[%.3f, %.3f]  std range=[%.3f, %.3f]",
                 float(joint_mean.min()), float(joint_mean.max()),
                 float(joint_std.min()), float(joint_std.max()))

    # ---- 5c. per-slide loop using shared _score_and_group_one_slide ----
    processed = []
    per_slide_meta = {}
    for sid, a in zip(used_ids, prepared):
        log.info("-" * 60)
        log.info("[multi.slide] %s  N=%d", sid, a.n_obs)

        # gene alignment for THIS slide
        var_pos = pd.Series(np.arange(a.n_vars), index=a.var_names)
        target_idx = var_pos.loc[shared_genes].values.astype(np.int64)

        if sid in target_X_cache:
            # batch_correction pre-pass already produced corrected target_X
            target_X = target_X_cache.pop(sid)
        else:
            # CPM normalise (per-slide, adaptive GPU)
            use_gpu_a = gpu_available() and a.n_obs >= cfg.graph.gpu_n_threshold
            target_X = normalize_target(a, target_idx,
                                        use_gpu=use_gpu_a, logger=log)

            # per-slide reconstruction (GraphVAE for spatial; cross-slide edges
            # are blocked inside build_spatial_graph via slide_id).
            if cfg.reconstruct:
                target_X = maybe_reconstruct(
                    adata=a, target_X=target_X, target_idx=target_idx,
                    data_mode="spatial", enabled=True, platform=platform,
                    spatial_radius_um=spatial_radius_um,
                    config=cfg, device=device, logger=log,
                )

        # z-score: joint mean/std or per-slide own
        if norm_mode == "joint":
            sd_safe = np.maximum(joint_std, 1e-6)
            target_X_z = np.clip(
                (target_X - joint_mean) / sd_safe, -10, 10,
            ).astype(np.float32)
        else:
            target_X_z, _, _ = zscore_columns(target_X, clip=10.0)
        del target_X
        gc.collect()

        # delegate to shared helper
        out = _score_and_group_one_slide(
            adata=a, target_X_z=target_X_z,
            teacher_res=teacher_res,
            bulk_pos_scores=bulk_pos_scores, bulk_neg_scores=bulk_neg_scores,
            cfg=cfg, data_mode="spatial", platform=platform,
            spatial_radius_um=spatial_radius_um,
            device=device, random_seed=random_seed, logger=log,
            multi_slide_mode=True,   # disables GNN distill (use pseudo_bulk only)
        )
        # target_X_z freed inside helper

        per_slide_meta[sid] = {
            "n_obs": int(a.n_obs),
            "spatial_used": bool(out["ran_spatial"]),
            "per_cell": out["per_cell"],
            "group_meta": out["group_meta"],
            "s_cell_mean": float(out["per_cell"]["s_cell"].mean()),
            "s_cell_std":  float(out["per_cell"]["s_cell"].std()),
        }
        groups = out["groups"]
        log.info("[multi.slide] %s done. groups=%s",
                 sid,
                 {k: int((groups == k).sum()) for k in
                  ["negative", "background", "positive"]})
        processed.append(a)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 6. Merge per-slide AnnData into one ----
    merged = ad.concat(
        processed, axis=0, join="inner", merge="same",
        label="slide_id", keys=used_ids, index_unique="-",
    )

    # rebuild per_cell arrays in the order of merged.obs
    pc_keys = list({k for v in per_slide_meta.values() for k in v["per_cell"]})
    merged_per_cell = {k: [] for k in pc_keys}
    for sid in used_ids:
        pc = per_slide_meta[sid]["per_cell"]
        for k in pc_keys:
            merged_per_cell[k].append(
                pc.get(k, np.zeros(per_slide_meta[sid]["n_obs"], dtype=np.float32))
            )
    merged_per_cell = {k: np.concatenate(v, axis=0) for k, v in merged_per_cell.items()}

    merged.uns["MAPPER_meta"] = {
        "version": "0.7",
        "data_mode": "spatial",
        "platform": platform,
        "n_slides": len(used_ids),
        "slide_ids": used_ids,
        "n_genes_shared": len(shared_genes),
        "refine_method": refine_method,
        "refine_alpha": float(cfg.niche.refine_alpha),
        "log_lr_threshold": tau,
        "teacher_metric": teacher_res.metric_name,
        "teacher_val_metric": float(teacher_res.val_metric),
        "phenotype_type": phenotype_type,
        "bulk_distribution_info": bulk_dist_info,
        "per_slide": {sid: {k: v for k, v in m.items() if k != "per_cell"}
                      for sid, m in per_slide_meta.items()},
        "per_cell": merged_per_cell,
        "normalization_mode": norm_mode,
    }
    log.info("[multi] DONE.  merged N=%d  slides=%d", merged.n_obs, len(used_ids))

    if return_concat:
        return merged
    out_list = []
    for sid in used_ids:
        sel = merged.obs["slide_id"] == sid
        out_list.append(merged[sel].copy())
    return out_list


# ---------------------------------------------------------------------------
# shared per-slide score + group helper (used by mapper and mapper_multi_slide)
# ---------------------------------------------------------------------------
def _score_and_group_one_slide(
    adata,
    target_X_z: np.ndarray,
    teacher_res,
    bulk_pos_scores,
    bulk_neg_scores,
    *,
    cfg: _cfg.MapperConfig,
    data_mode: str,
    platform: str,
    spatial_radius_um: Optional[float],
    device: torch.device,
    random_seed: int,
    logger,
    multi_slide_mode: bool = False,
) -> dict:
    """Score + group a single slide given (pre-trained) teacher + KDE refs.

    Pre-conditions
    --------------
    * ``adata`` carries the spatial coords (``obsm['spatial']`` if spatial mode).
    * ``target_X_z`` is already normalised + z-scored by the caller
      (single-slide ``mapper`` uses per-slide z-score; multi-slide uses joint
      or per-slide depending on ``cfg.multi_slide_normalization``).
    * ``teacher_res.model`` and the bulk KDE arrays are computed once and
      passed in so this function is purely inference + grouping.

    Side-effects
    ------------
    * Writes ``adata.obs['MAPPER.Score']`` and ``adata.obs['MAPPER.Group']``.
    """
    log = logger or logging.getLogger("MAPPER")
    distributions_ok = bulk_pos_scores is not None and bulk_neg_scores is not None

    # 1. Teacher inference
    teacher_res.model.eval()
    s_cell_raw, teacher_emb = _teacher_infer_chunked(
        teacher_res.model, target_X_z,
        devices=list_cuda_devices() if gpu_available() else [device],
        batch_size=cfg.teacher_inference_batch_size,
        logger=log,
    )
    log.info("[score] s_cell_raw: mean=%.3f std=%.3f min=%.3f max=%.3f",
             float(np.nanmean(s_cell_raw)), float(np.nanstd(s_cell_raw)),
             float(s_cell_raw.min()), float(s_cell_raw.max()))

    # 2. Expression PCA for scRNA Leiden grouping
    expr_pca = None
    if data_mode == "scRNA":
        expr_pca = _expression_pca(
            target_X_z, n_components=cfg.grouping.pca_dim, logger=log,
        )

    # 3. Spatial niche (spatial mode only)
    ran_spatial = False
    s_niche_raw = np.zeros_like(s_cell_raw)
    refinement = "none"
    if data_mode == "spatial" and is_spatial_platform(platform) \
            and cfg.niche.refine_method != "none":
        radius_um = resolve_default_radius_um(platform, spatial_radius_um)
        if multi_slide_mode:
            # coords were pre-converted to micron in mapper_multi_slide step 1
            coords_um = adata.obsm.get("spatial", None)
        else:
            # single-slide: convert pixel → micron via uns["spatial"] scalefactors
            coords_um = coords_to_micron(adata, platform, logger=log)
        if coords_um is None or radius_um is None or radius_um <= 0:
            log.warning("[score] coords/radius unavailable; spatial step skipped")
        else:
            slide_ids = None
            if not multi_slide_mode and "slide_id" in adata.obs.columns:
                slide_ids = adata.obs["slide_id"].to_numpy()
            G_spatial = build_spatial_graph(
                np.asarray(coords_um, dtype=np.float32), radius_um,
                max_neighbors=cfg.graph.max_neighbors,
                slide_ids=slide_ids,
                chunk=cfg.graph.spatial_chunk_size,
                use_torch_cluster=detect_backends().torch_cluster,
                device=device, logger=log,
            )
            if G_spatial is not None:
                ran_spatial = True
                s_niche_raw, _ = compute_s_niche_direct(
                    target_X_z, G_spatial, teacher_res.model,
                    edge_weight=None,
                    include_self=cfg.niche.include_self_in_pseudobulk,
                    device=device,
                    batch_size=cfg.teacher_inference_batch_size,
                    logger=log,
                )
                # GNN distillation works per-slide in both single and multi
                # (each slide trains its own SAGE on its G_spatial + teacher_emb,
                # distilling its own pseudo-bulk niche label).
                if cfg.niche.refine_method == "gnn":
                    free_gpu_memory(logger=log)
                    student_device = pick_best_cuda_device(logger=log)
                    s_niche_raw = fit_student_distill(
                        teacher_emb, s_niche_raw, G_spatial,
                        config=cfg.student, device=student_device,
                        random_seed=random_seed, logger=log,
                    )
                    refinement = "gnn_distill"
                else:
                    refinement = "pseudo_bulk"

    # 4. Composite score
    alpha = float(cfg.niche.refine_alpha) if ran_spatial else 0.0
    s_final = ((1.0 - alpha) * s_cell_raw + alpha * s_niche_raw).astype(np.float32)

    # 5. Likelihood ratios
    if distributions_ok:
        log_lr_cell = log_likelihood_ratio(
            s_cell_raw, bulk_pos_scores, bulk_neg_scores,
            bandwidth=cfg.niche.kde_bandwidth,
            clip=cfg.niche.log_lr_clip, logger=log,
        )
        log_lr_niche = (
            log_likelihood_ratio(
                s_niche_raw, bulk_pos_scores, bulk_neg_scores,
                bandwidth=cfg.niche.kde_bandwidth,
                clip=cfg.niche.log_lr_clip, logger=log,
            ) if ran_spatial else np.zeros_like(s_niche_raw)
        )
    else:
        log_lr_cell = np.zeros_like(s_cell_raw)
        log_lr_niche = np.zeros_like(s_niche_raw)

    # 6. Grouping
    tau = float(cfg.niche.log_lr_threshold)
    bulk_P = int(bulk_pos_scores.size) if bulk_pos_scores is not None else None
    bulk_N = int(bulk_neg_scores.size) if bulk_neg_scores is not None else None
    if data_mode == "scRNA":
        if expr_pca is None or not distributions_ok:
            log.warning("[score] expr_pca / distributions unavailable; cell-only fallback")
            groups, group_meta = likelihood_ratio_3class(
                log_lr_cell, None, log_lr_threshold=tau,
                bulk_pos_size=bulk_P, bulk_neg_size=bulk_N, logger=log,
            )
        else:
            groups, group_meta = cluster_phenotype_assign_lr(
                log_lr_cell, expr_pca,
                log_lr_threshold=tau,
                bulk_pos_size=bulk_P, bulk_neg_size=bulk_N,
                config=cfg.grouping, random_seed=random_seed, logger=log,
            )
    else:
        groups, group_meta = likelihood_ratio_3class(
            log_lr_cell, log_lr_niche if ran_spatial else None,
            log_lr_threshold=tau,
            bulk_pos_size=bulk_P, bulk_neg_size=bulk_N, logger=log,
        )

    # 7. Write per-slide obs (2 columns only)
    adata.obs["MAPPER.Score"] = np.asarray(s_final, dtype=float)
    adata.obs["MAPPER.Group"] = pd.Categorical(
        groups, categories=["negative", "background", "positive"], ordered=True,
    )

    per_cell = {
        "s_cell":  np.asarray(s_cell_raw, dtype=np.float32),
        "LR_cell": np.asarray(log_lr_cell, dtype=np.float32),
    }
    if ran_spatial:
        per_cell["s_niche"]  = np.asarray(s_niche_raw, dtype=np.float32)
        per_cell["LR_niche"] = np.asarray(log_lr_niche, dtype=np.float32)

    return {
        "ran_spatial": ran_spatial,
        "refinement": refinement,
        "refine_alpha": alpha,
        "s_final": s_final,
        "groups": groups,
        "group_meta": group_meta,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _teacher_infer_chunked(model, X, *, devices, batch_size, logger=None):
    log = logger or logging.getLogger("MAPPER")
    N, G = X.shape
    bs = int(batch_size)
    devices = list(devices) if devices else [auto_device()]

    per_dev_model = []
    for d in devices:
        m = copy.deepcopy(model).to(d).eval()
        for p in m.parameters():
            p.requires_grad = False
        per_dev_model.append(m)

    emb_dim = int(model.head.in_features)
    s_raw = np.empty(N, dtype=np.float32)
    emb = np.empty((N, emb_dim), dtype=np.float32)
    starts = list(range(0, N, bs))
    log.info("[teacher] target inference: N=%d G=%d chunks=%d batch=%d devices=%s",
             N, G, len(starts), bs, [str(d) for d in devices])

    def _process(chunk_id):
        s = starts[chunk_id]
        e = min(s + bs, N)
        gpu_idx = chunk_id % len(devices)
        d = devices[gpu_idx]
        m = per_dev_model[gpu_idx]
        with torch.no_grad():
            x_chunk = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(d, non_blocking=True)
            out, z = m(x_chunk)
            s_raw[s:e] = out.view(-1).detach().to("cpu", non_blocking=False).numpy()
            emb[s:e] = z.detach().to("cpu", non_blocking=False).numpy()
            del x_chunk, out, z

    if len(devices) <= 1:
        for cid in range(len(starts)):
            _process(cid)
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            for _ in ex.map(_process, range(len(starts))):
                pass

    for m in per_dev_model:
        del m
    per_dev_model.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return s_raw, emb


def _expression_pca(X, *, n_components=50, logger=None):
    log = logger or logging.getLogger("MAPPER")
    try:
        from sklearn.decomposition import TruncatedSVD
        n_comp = int(min(n_components, X.shape[1] - 1, X.shape[0] - 1))
        log.info("[expr_pca] TruncatedSVD: %s -> %d components", X.shape, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        out = svd.fit_transform(X).astype(np.float32)
        log.info("[expr_pca] explained variance ratio sum=%.3f",
                 float(svd.explained_variance_ratio_.sum()))
        return out
    except Exception as e:
        log.warning("[expr_pca] failed (%s); using raw matrix.", e)
        return np.asarray(X, dtype=np.float32)
