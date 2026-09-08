from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import axes as ax
from . import denoise as dn
from . import pergene
from .calling import BACKGROUND, anchor_to_cohort, call, reference_scores
from .config import MapperConfig
from .geometry import check_neighbourhood, neighbour_graph, niche_profile, um_per_unit
from .io_slide import detection_fraction, read_slide_matrix
from .panel import build_panel
from .teacher import fit_teacher
from .uns_log import UnsLog

SCORE_COLUMN = "MAPPER.Score"
GROUP_COLUMN = "MAPPER.Group"
COMPOSITION_TO_TPM = 1e6

MIN_COUNTS_PER_UNIT = 10

EXPECTED_NEIGHBOURS = {"visium": (4, 8), "visium_hd": (5, 60), "xenium": (5, 60),
                       "atera": (5, 60), "openst": (5, 60)}

@dataclass(frozen=True)
class Representation:
    unit: np.ndarray
    niche: np.ndarray
    has_niche: np.ndarray
    cohort: np.ndarray
    pooled: np.ndarray
    axis_names: list
    genes: np.ndarray
    library: np.ndarray

    sequencing_library: np.ndarray | None

    adj: object
    usable: np.ndarray
    resolution: str
    adata: object
    log: UnsLog

@dataclass(frozen=True)
class MapperResult:
    score: np.ndarray
    group: np.ndarray
    uns: dict

def _rownorm(x):
    return x / np.maximum(x.sum(axis=-1, keepdims=True), 1e-300)

def prepare(adata, cohort, platform: str, cfg: MapperConfig | None = None,
            section_key=None, counts_layer=None, gene_sets=None,
            allow_normalized_surrogate: bool = False, coord_key: str = "spatial",
            log=None) -> Representation:
    cfg = cfg or MapperConfig()
    log = log or UnsLog()
    log.record("version", "0.4.0")
    log.record("platform", platform)
    log.record("seed", cfg.seed)

    slide = read_slide_matrix(adata, counts_layer=counts_layer, log=log,
                              allow_normalized_surrogate=allow_normalized_surrogate)
    panel = build_panel(slide.genes, cohort.genes, detection_fraction(slide.counts),
                        cohort.expr.mean(axis=0) * COMPOSITION_TO_TPM, cfg, log=log)
    counts = sparse.csr_matrix(slide.counts)[:, panel.slide_index]
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    usable = library >= MIN_COUNTS_PER_UNIT
    log.record("min_counts_per_unit", MIN_COUNTS_PER_UNIT, section="panel")
    log.record("n_units_below_depth_floor", int((~usable).sum()), section="panel")
    if not usable.any():
        raise ValueError(f"no unit reaches {MIN_COUNTS_PER_UNIT} panel counts")
    counts, library = counts[usable], library[usable]

    q = cfg.extra.get("match_unit_depth")
    if q is None:
        log.record("unit_depth_matched_to", None, section="panel")
        log.record("unit_depth_quantile", None, section="panel")
    else:
        target = int(np.quantile(library, float(q)))
        counts = ax.thin_counts(counts, library, target, seed=cfg.seed)
        library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        log.record("unit_depth_matched_to", target, section="panel")
        log.record("unit_depth_quantile", float(q), section="panel")
        log.record("n_units_already_shallower", int((library < target).sum()),
                   section="panel")
    bulk_p = _rownorm(cohort.expr[:, panel.bulk_index])

    resolution = cfg.resolution(platform)
    log.record("resolution", resolution)

    adj = _graph(adata, platform, cfg, section_key, usable, coord_key, log)
    mode = str(cfg.extra.get("denoise") or "none")
    if mode not in dn.MODES:
        raise ValueError(f"unknown denoise mode {mode!r}; expected one of {dn.MODES}")
    if mode != "none":
        if adj is None:
            raise ValueError(
                f"platform {platform!r} has no neighbourhood, so there is "
                f"nothing to denoise against"
            )
        if resolution != "spot":
            raise ValueError(
                f"denoising works on the dense per-gene matrix and so is a "
                f"spot-resolution step; {platform!r} is read on the gene-set "
                f"axes at cell resolution"
            )
        dn.check_dense_budget(counts.shape[0], counts.shape[1])

    depth = int(np.median(library))

    pooled_p = _rownorm(np.asarray(counts.sum(axis=0), dtype=np.float64).reshape(1, -1))

    if resolution == "spot":
        space = pergene.build(panel.genes, bulk_p, target_depth=depth,
                              seed=cfg.seed, log=log)
        if mode == "none":
            log.record("mode", "none", section="denoise")
            log.record("reproducible", True, section="denoise")
            unit = pergene.score_units(counts, library, space)
        else:

            unit = pergene.score_units_from_log1p(
                dn.denoise(mode, pergene.log1p_cp10k(counts, library), adj,
                           library, seed=cfg.seed, log=log),
                space)
        cohort_z = pergene.cohort_scores(space)
        names = list(panel.genes)
        pooled = np.clip(
            (ax.depth_matched_log1p(pooled_p, depth, seed=cfg.seed) - space.mean)
            / space.sd, -space.clip, space.clip)
    else:
        space = ax.build(panel.genes, gene_sets or ax.load_gene_sets(), bulk_p,
                         min_genes=cfg.hallmark_min_genes,
                         target_depth=depth, seed=cfg.seed, log=log)
        keep, _ = ax.admissible(counts, space, cfg.reliability_floor(resolution),
                                seed=cfg.seed, log=log)
        if not keep.all():
            space = ax.restrict(space, keep)
        log.record("mode", "none", section="denoise")
        log.record("reproducible", True, section="denoise")
        unit = ax.score_units(counts, library, space)
        cohort_z = (space.cohort_raw - space.mean) / space.sd
        names = list(space.names)
        pooled = (ax.depth_matched_bulk(pooled_p, space.weights, depth,
                                        seed=cfg.seed) - space.mean) / space.sd
    if adj is None:
        niche = np.full_like(unit, np.nan)
        has = np.zeros(unit.shape[0], dtype=bool)
    else:
        niche, has = niche_profile(unit, adj, library)
    seq = slide.sequencing_library
    return Representation(unit=unit, niche=niche, has_niche=has,
                          cohort=cohort_z, pooled=pooled,
                          axis_names=names, genes=panel.genes,
                          library=library,
                          sequencing_library=None if seq is None else seq[usable],
                          adj=adj, usable=usable, resolution=resolution,
                          adata=adata, log=log)

def _graph(adata, platform, cfg, section_key, usable, coord_key, log):
    radius = cfg.radius_um(platform)
    if radius is None:
        log.record("niche", "disabled: scRNA has no neighbourhood", section="geometry")
        return None
    if coord_key not in adata.obsm:
        raise ValueError(
            f"platform {platform!r} needs obsm[{coord_key!r}], which is absent; "
            f"this slide carries {sorted(adata.obsm)}"
        )
    xy = np.asarray(adata.obsm[coord_key], dtype=float)
    if xy.ndim != 2 or xy.shape[1] not in (2, 3):
        raise ValueError(
            f"obsm[{coord_key!r}] must be (n, 2) or (n, 3); got {xy.shape}. "
            f"Trimming it here would change the physical neighbourhood without "
            f"saying so"
        )
    xy = xy[usable]
    section = None
    if section_key is not None:
        section = np.asarray(adata.obs[section_key])[usable]
        log.record("section_key", section_key, section="geometry")
    log.record("coord_key", coord_key, section="geometry")
    log.record("coord_dimensions", int(xy.shape[1]), section="geometry")
    scale = um_per_unit(xy, platform, adata.uns.get("spatial"), log=log,
                        section=section)
    adj = neighbour_graph(xy, radius, scale, section=section)
    check_neighbourhood(adj, radius, EXPECTED_NEIGHBOURS.get(platform, (1, 10_000)), log)
    return adj

def apply_head(rep: Representation, cohort, cfg: MapperConfig | None = None):
    cfg = cfg or MapperConfig()
    log = rep.log.copy()
    clinical = _clinical(cohort)
    kind = str(cfg.extra.get("teacher", "mlp"))

    teacher = fit_teacher(rep.cohort, clinical, cohort.endpoint, kind=kind,
                          seed=cfg.seed, log=log)

    frame = str(cfg.extra.get("reference_frame", "offset"))
    hi, lo, sign = reference_scores(teacher, rep.cohort, clinical,
                                    cohort.endpoint, frame=frame, log=log)

    unit_score = sign * teacher.score(rep.unit)
    niche_score = np.full(unit_score.shape, np.nan)
    if rep.has_niche.any():
        niche_score[rep.has_niche] = sign * teacher.score(rep.niche[rep.has_niche])
    pooled_score = float(np.ravel(sign * teacher.score(rep.pooled))[0])

    anchored, _, shift = anchor_to_cohort(unit_score, niche_score, pooled_score,
                                          log=log)
    log.record("unit_mean", round(float(np.nanmean(unit_score)), 6), section="head")
    if bool(cfg.extra.get("anchor", False)):
        unit_score, niche_score = anchored, niche_score - shift
        log.record("anchor_applied", True, section="head")
        log.record("anchored_unit_mean", round(float(np.nanmean(unit_score)), 6),
                   section="head")
    else:
        log.record("anchor_applied", False, section="head")

    if frame == "offset":
        slide_centre = float(np.nanmedian(unit_score))
        slide_spread = float(np.nanstd(unit_score)) or 1.0
        hi = slide_centre + hi * slide_spread
        lo = slide_centre + lo * slide_spread
        log.record("slide_centre", round(slide_centre, 4), section="calls")
        log.record("slide_spread", round(slide_spread, 4), section="calls")
        log.record("level_bad_on_slide", round(hi, 4), section="calls")
        log.record("level_good_on_slide", round(lo, 4), section="calls")

    a = float(cfg.extra.get("niche_weight", 0.5))
    log.record("niche_weight", a, section="head")
    combined = np.where(np.isfinite(niche_score),
                        (1.0 - a) * unit_score + a * niche_score, unit_score)

    group = call(unit_score, niche_score, hi, lo,
                 rule=str(cfg.extra.get("call_rule", "mean")),
                 niche_weight=a, log=log)

    finite = np.isfinite(combined)
    if finite.sum() > 2:
        from scipy.stats import spearmanr

        true_depth = (rep.library if rep.sequencing_library is None
                      else rep.sequencing_library)
        log.record("score_library_spearman",
                   round(float(spearmanr(combined[finite],
                                         true_depth[finite]).statistic), 4),
                   section="head")
        log.record("score_library_is_sequencing_depth",
                   rep.sequencing_library is not None, section="head")
        if rep.sequencing_library is not None:
            log.record("score_corrected_library_spearman",
                       round(float(spearmanr(combined[finite],
                                             rep.library[finite]).statistic), 4),
                       section="head")
    else:
        log.record("score_library_spearman", float("nan"), section="head")
        log.record("score_library_is_sequencing_depth", False, section="head")

    full_score = np.full(rep.usable.size, np.nan)
    full_group = np.full(rep.usable.size, BACKGROUND, dtype=object)
    full_score[rep.usable] = combined
    full_group[rep.usable] = group
    rep.adata.obs[SCORE_COLUMN] = full_score
    rep.adata.obs[GROUP_COLUMN] = full_group
    rep.adata.uns["MAPPER"] = log.as_dict()
    return MapperResult(score=full_score, group=full_group,
                        uns=rep.adata.uns["MAPPER"])

def _clinical(cohort) -> dict:
    if cohort.endpoint == "survival":
        return {"time": np.asarray(cohort.time), "event": np.asarray(cohort.event)}
    if cohort.endpoint == "classification":
        return {"label": np.asarray(cohort.label)}
    return {"value": np.asarray(cohort.value)}

def run(adata, cohort, platform: str, cfg=None, **kw):
    return apply_head(prepare(adata, cohort, platform, cfg, **kw), cohort, cfg)
