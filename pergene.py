from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

_EPS = 1e-12

_MIN_SD = 1e-8

DEFAULT_CLIP = 10.0

CHUNK = 20_000

@dataclass(frozen=True)
class GeneSpace:
    mean: np.ndarray
    sd: np.ndarray
    clip: float
    cohort_log1p: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.mean.size)

def build(genes, cohort_composition, clip: float = DEFAULT_CLIP,
          target_depth: int | None = None, seed: int = 42, log=None) -> GeneSpace:
    comp = np.atleast_2d(np.asarray(cohort_composition, dtype=np.float64))
    if target_depth is None:
        b = np.log1p(1e4 * comp)
    else:
        from .axes import depth_matched_log1p
        b = depth_matched_log1p(comp, int(target_depth), seed=seed)
    mean, sd = b.mean(axis=0), b.std(axis=0)
    flat = sd < _MIN_SD
    space = GeneSpace(mean=mean, sd=np.where(flat, 1.0, sd), clip=float(clip),
                      cohort_log1p=b)
    if log is not None:
        log.record("coordinate", "per-gene", section="axes")
        log.record("n_genes", int(mean.size), section="axes")
        log.record("n_genes_constant_in_cohort", int(flat.sum()), section="axes")
        log.record("clip", float(clip), section="axes")
        log.record("standardiser", "cohort, applied to both domains", section="axes")
    return space

def score_bulk(cohort_composition, space: GeneSpace) -> np.ndarray:
    b = np.log1p(1e4 * np.atleast_2d(np.asarray(cohort_composition, dtype=np.float64)))
    return np.clip((b - space.mean) / space.sd, -space.clip, space.clip)

def cohort_scores(space: GeneSpace) -> np.ndarray:
    if space.cohort_log1p is None:
        raise ValueError("this GeneSpace kept no cohort readings")
    return np.clip((space.cohort_log1p - space.mean) / space.sd,
                   -space.clip, space.clip)

def _block(counts, library, space, start, stop) -> np.ndarray:
    block = counts[start:stop]
    scale = 1e4 / np.maximum(np.asarray(library[start:stop], float), _EPS)
    dense = np.zeros(block.shape, dtype=np.float64)
    if block.nnz:
        rows = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
        dense[rows, block.indices] = np.log1p(block.data * scale[rows])
    return np.clip((dense - space.mean) / space.sd, -space.clip, space.clip)

def score_units(counts, library, space: GeneSpace, chunk: int = CHUNK) -> np.ndarray:
    counts = sparse.csr_matrix(counts)
    out = np.empty(counts.shape, dtype=np.float32)
    for start in range(0, counts.shape[0], chunk):
        stop = min(start + chunk, counts.shape[0])
        out[start:stop] = _block(counts, library, space, start, stop)
    return out

def log1p_cp10k(counts, library, chunk: int = CHUNK) -> np.ndarray:
    counts = sparse.csr_matrix(counts)
    library = np.asarray(library, dtype=float).ravel()
    out = np.zeros(counts.shape, dtype=np.float32)
    for start in range(0, counts.shape[0], chunk):
        stop = min(start + chunk, counts.shape[0])
        block = counts[start:stop]
        if not block.nnz:
            continue
        scale = 1e4 / np.maximum(library[start:stop], _EPS)
        rows = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
        out[start + rows, block.indices] = np.log1p(block.data * scale[rows])
    return out

def score_units_from_log1p(dense_log1p, space: GeneSpace) -> np.ndarray:
    x = np.asarray(dense_log1p, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != space.mean.size:
        raise ValueError(
            f"matrix has {x.shape[-1] if x.ndim == 2 else '?'} genes for a "
            f"space of {space.mean.size}"
        )
    return np.clip((x - space.mean) / space.sd,
                   -space.clip, space.clip).astype(np.float32)

def project(counts, library, space: GeneSpace, beta: np.ndarray,
            chunk: int = CHUNK) -> np.ndarray:
    counts = sparse.csr_matrix(counts)
    beta = np.asarray(beta, dtype=np.float64).ravel()
    if beta.size != len(space):
        raise ValueError(f"beta has {beta.size} entries for {len(space)} genes")
    out = np.empty(counts.shape[0])
    for start in range(0, counts.shape[0], chunk):
        stop = min(start + chunk, counts.shape[0])
        out[start:stop] = _block(counts, library, space, start, stop) @ beta
    return out
