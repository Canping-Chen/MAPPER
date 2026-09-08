from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

MT_PREFIX = "MT-"

@dataclass(frozen=True)
class Panel:

    genes: np.ndarray
    slide_index: np.ndarray
    bulk_index: np.ndarray

    def __len__(self) -> int:
        return int(self.genes.shape[0])

def _positions(names: np.ndarray, what: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    duplicates: list[str] = []
    for i, name in enumerate(names):
        key = str(name)
        if key in positions:
            if key not in duplicates:
                duplicates.append(key)
            continue
        positions[key] = i
    if duplicates:
        raise ValueError(
            f"{what} gene axis has duplicate symbols {duplicates[:5]}; which "
            f"column is which gene is ambiguous, so de-duplicate upstream"
        )
    return positions

def _is_mitochondrial(symbol: str) -> bool:
    return symbol.upper().startswith(MT_PREFIX)

TECHNICAL_RNA = re.compile(
    r"^(MTRNR"
    r"|RN7SL|RN7SK"
    r"|RNU[0-9]|RNVU"
    r"|RMRP$|RPPH1$"
    r"|SNORD|SNORA|SNORC|SCARNA"
    r"|RNA5S|RNA5-8S|RNA18S|RNA28S"
    r"|MIR[0-9])",
    re.IGNORECASE,
)

def _is_technical_rna(symbol: str) -> bool:
    return bool(TECHNICAL_RNA.match(symbol))

def build_panel(slide_genes, bulk_genes, detection, bulk_mean_tpm, cfg, log=None) -> Panel:
    slide_genes = np.asarray(slide_genes, dtype=object)
    bulk_genes = np.asarray(bulk_genes, dtype=object)
    detection = np.asarray(detection, dtype=float).ravel()
    bulk_mean_tpm = np.asarray(bulk_mean_tpm, dtype=float).ravel()
    if detection.shape[0] != slide_genes.shape[0]:
        raise ValueError(
            f"detection has {detection.shape[0]} entries but there are "
            f"{slide_genes.shape[0]} slide genes"
        )
    if bulk_mean_tpm.shape[0] != bulk_genes.shape[0]:
        raise ValueError(
            f"bulk_mean_tpm has {bulk_mean_tpm.shape[0]} entries but there are "
            f"{bulk_genes.shape[0]} bulk genes"
        )

    slide_pos = _positions(slide_genes, "slide")
    bulk_pos = _positions(bulk_genes, "bulk")

    kept = sorted(set(slide_pos) & set(bulk_pos))
    n_intersection = len(kept)

    kept = [g for g in kept if detection[slide_pos[g]] >= cfg.min_detection_fraction]
    n_after_detection = len(kept)

    kept = [g for g in kept if bulk_mean_tpm[bulk_pos[g]] >= cfg.bulk_mean_tpm_floor]
    n_after_bulk_floor = len(kept)

    mass_before = float(sum(bulk_mean_tpm[bulk_pos[g]] for g in kept))
    technical = [g for g in kept if _is_technical_rna(g)]
    if cfg.drop_technical_rna:
        kept = [g for g in kept if not _is_technical_rna(g)]
    mt = [g for g in kept if _is_mitochondrial(g)]
    mt_mass = float(sum(bulk_mean_tpm[bulk_pos[g]] for g in mt))
    if cfg.drop_mt:
        dropped = set(mt)
        kept = [g for g in kept if g not in dropped]
    else:
        mt, mt_mass = [], 0.0
    n_after_mt = len(kept)

    panel = Panel(
        genes=np.asarray(kept, dtype=object),
        slide_index=np.asarray([slide_pos[g] for g in kept], dtype=int),
        bulk_index=np.asarray([bulk_pos[g] for g in kept], dtype=int),
    )

    if log is not None:
        cohort_mass = float(bulk_mean_tpm.sum())
        cohort_mt = float(bulk_mean_tpm[
            [i for g, i in bulk_pos.items() if _is_mitochondrial(g)]
        ].sum()) if cohort_mass > 0 else 0.0
        log.record("n_slide_genes", int(slide_genes.shape[0]), section="panel")
        log.record("n_bulk_genes", int(bulk_genes.shape[0]), section="panel")
        log.record("n_intersection", n_intersection, section="panel")
        log.record("n_after_detection", n_after_detection, section="panel")
        log.record("n_after_bulk_floor", n_after_bulk_floor, section="panel")
        log.record("n_after_mt", n_after_mt, section="panel")
        log.record("n_panel", len(panel), section="panel")
        log.record("mt_dropped", len(mt), section="panel")
        log.record("mt_bulk_mass_share",
                   mt_mass / mass_before if mass_before > 0 else 0.0, section="panel")
        log.record("mt_bulk_mass_share_cohort",
                   cohort_mt / cohort_mass if cohort_mass > 0 else 0.0, section="panel")
        log.record("min_detection_fraction", float(cfg.min_detection_fraction),
                   section="panel")
        log.record("bulk_mean_tpm_floor", float(cfg.bulk_mean_tpm_floor), section="panel")
        log.record("drop_mt", bool(cfg.drop_mt), section="panel")
        log.record("drop_technical_rna", bool(cfg.drop_technical_rna), section="panel")
        log.record("n_technical_rna_dropped",
                   int(len(technical)) if cfg.drop_technical_rna else 0, section="panel")
        tech_mass = float(sum(bulk_mean_tpm[bulk_pos[g]] for g in technical))
        log.record("technical_rna_bulk_mass_share",
                   tech_mass / mass_before if mass_before > 0 else 0.0, section="panel")

    attrition = (
        f"{slide_genes.shape[0]} slide / {bulk_genes.shape[0]} bulk genes -> "
        f"{n_intersection} shared -> {n_after_detection} detected -> "
        f"{n_after_bulk_floor} above the bulk floor -> {n_after_mt} after MT"
    )
    if len(panel) < cfg.panel_error_below:
        raise ValueError(
            f"panel has {len(panel)} genes, below panel_error_below="
            f"{cfg.panel_error_below}; nothing can be estimated across domains "
            f"on this few. Attrition: {attrition}"
        )
    if len(panel) < cfg.panel_warn_below:
        warnings.warn(
            f"panel has {len(panel)} genes, below panel_warn_below="
            f"{cfg.panel_warn_below}; cross-domain estimates will be noisy. "
            f"Attrition: {attrition}",
            UserWarning,
            stacklevel=2,
        )
    return panel
