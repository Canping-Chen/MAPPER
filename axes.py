from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

_EPS = 1e-12
_RESOURCES = Path(__file__).resolve().parent / "resources"

_MIN_SD = 1e-8

CHUNK = 50_000

def _read_sets(path: Path) -> dict:
    if path.suffix == ".tsv":
        import pandas as pd
        df = pd.read_csv(path, sep="\t")
        df["gs_name"] = df["gs_name"].str.replace("HALLMARK_", "", regex=False)
        return {n: set(g["gene_symbol"]) for n, g in df.groupby("gs_name")}
    return {k: set(v) for k, v in json.loads(path.read_text()).items()}

def load_gene_sets(path=None) -> dict:
    return _read_sets(Path(path) if path is not None
                      else _RESOURCES / "hallmark.json")

@dataclass(frozen=True)
class AxisSpace:

    weights: sparse.csr_matrix
    names: list
    mean: np.ndarray
    sd: np.ndarray
    cohort_raw: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.names)

def weight_matrix(genes, gene_sets, min_genes: int = 5):
    genes = np.asarray(genes, dtype=object)
    position = {str(g): i for i, g in enumerate(genes)}
    rows, cols, data, names = [], [], [], []
    for name in sorted(gene_sets):
        members = [position[str(g)] for g in gene_sets[name] if str(g) in position]
        if len(members) < min_genes:
            continue
        idx = len(names)
        names.append(name)
        rows.extend(members)
        cols.extend([idx] * len(members))
        data.extend([1.0 / len(members)] * len(members))
    if not names:
        raise ValueError(
            f"no gene set has at least {min_genes} members on a panel of "
            f"{genes.size} genes; the panel is too narrow for set-based axes"
        )
    return sparse.csr_matrix((data, (rows, cols)), shape=(genes.size, len(names))), names

def raw_scores(counts, library, weights, chunk: int = CHUNK) -> np.ndarray:
    counts = sparse.csr_matrix(counts)
    library = np.asarray(library, dtype=float).ravel()
    if library.size != counts.shape[0]:
        raise ValueError(f"library has {library.size} entries for {counts.shape[0]} units")
    weights = sparse.csr_matrix(weights)
    out = np.empty((counts.shape[0], weights.shape[1]), dtype=np.float32)
    for start in range(0, counts.shape[0], chunk):
        stop = min(start + chunk, counts.shape[0])
        block = counts[start:stop]
        scale = 1e4 / np.maximum(library[start:stop], _EPS)
        block = sparse.csr_matrix(
            (np.log1p(block.data * np.repeat(scale, np.diff(block.indptr))),
             block.indices, block.indptr), shape=block.shape)
        out[start:stop] = (block @ weights).toarray()
    return out

def bulk_raw_scores(composition, weights) -> np.ndarray:
    p = np.atleast_2d(np.asarray(composition, dtype=np.float64))
    w = np.asarray(sparse.csr_matrix(weights).todense(), dtype=np.float64)
    return np.log1p(1e4 * p) @ w

DEPTH_MATCH_DRAWS = 10

def thin_counts(counts, library, depth: int, seed: int = 42):
    counts = sparse.csr_matrix(counts)
    library = np.asarray(library, dtype=float).ravel()
    rng = np.random.default_rng(int(seed))
    keep = np.minimum(float(depth) / np.maximum(library, _EPS), 1.0)
    per_value = np.repeat(keep, np.diff(counts.indptr))
    data = rng.binomial(counts.data.astype(np.int64), per_value).astype(np.float64)
    out = sparse.csr_matrix((data, counts.indices.copy(), counts.indptr.copy()),
                            shape=counts.shape)
    out.eliminate_zeros()
    return out

def depth_matched_log1p(composition, depth: int, draws: int = DEPTH_MATCH_DRAWS,
                        seed: int = 42) -> np.ndarray:
    p = np.atleast_2d(np.asarray(composition, dtype=np.float64))
    p = p / np.maximum(p.sum(axis=1, keepdims=True), _EPS)
    rng = np.random.default_rng(int(seed))
    total = None
    for _ in range(int(draws)):
        counts = np.vstack([rng.multinomial(int(depth), row) for row in p])
        got = np.log1p(1e4 * counts / float(depth))
        total = got if total is None else total + got
    return total / float(draws)

def depth_matched_bulk(composition, weights, depth: int, draws: int = DEPTH_MATCH_DRAWS,
                       seed: int = 42) -> np.ndarray:
    lg = depth_matched_log1p(composition, depth, draws=draws, seed=seed)
    w = np.asarray(sparse.csr_matrix(weights).todense(), dtype=np.float64)
    return lg @ w

def build(genes, gene_sets, cohort_composition, min_genes: int = 5,
          target_depth: int | None = None, seed: int = 42, log=None) -> AxisSpace:
    weights, names = weight_matrix(genes, gene_sets, min_genes=min_genes)
    if target_depth is None:
        raw = bulk_raw_scores(cohort_composition, weights)
    else:
        raw = depth_matched_bulk(cohort_composition, weights, int(target_depth), seed=seed)
    mean, sd = raw.mean(axis=0), raw.std(axis=0)
    flat = sd < _MIN_SD
    space = AxisSpace(weights=weights, names=list(names), mean=mean,
                      sd=np.where(flat, 1.0, sd), cohort_raw=raw)
    if log is not None:
        log.record("coordinate", "hallmark", section="axes")
        log.record("n_axes", len(names), section="axes")
        log.record("axis_names", list(names), section="axes")
        log.record("n_axes_constant_in_cohort", int(flat.sum()), section="axes")
        log.record("standardiser", "cohort, applied to both domains", section="axes")
        log.record("depth_matched_to", None if target_depth is None else int(target_depth),
                   section="axes")
        if target_depth is not None:
            unmatched = bulk_raw_scores(cohort_composition, weights)
            log.record("depth_bias_removed", round(float((raw - unmatched).mean()), 4),
                       section="axes")
    return space

def score_units(counts, library, space: AxisSpace, chunk: int = CHUNK) -> np.ndarray:
    return (raw_scores(counts, library, space.weights, chunk=chunk).astype(float)
            - space.mean) / space.sd

def score_bulk(cohort_composition, space: AxisSpace) -> np.ndarray:
    return (bulk_raw_scores(cohort_composition, space.weights) - space.mean) / space.sd

def reliability(counts, weights, seed: int = 42, chunk: int = CHUNK) -> np.ndarray:
    counts = sparse.csr_matrix(counts)
    weights = sparse.csr_matrix(weights)
    rng = np.random.default_rng(seed)
    n_axes = weights.shape[1]
    a = np.empty((counts.shape[0], n_axes), dtype=np.float32)
    b = np.empty_like(a)
    for start in range(0, counts.shape[0], chunk):
        stop = min(start + chunk, counts.shape[0])
        block = counts[start:stop]
        first = rng.binomial(block.data.astype(np.int64), 0.5).astype(np.float64)
        for out, data in ((a, first), (b, block.data - first)):
            lib = np.add.reduceat(data, block.indptr[:-1]) if block.shape[0] else data
            lib[np.diff(block.indptr) == 0] = 0.0
            scaled = sparse.csr_matrix(
                (np.log1p(data * np.repeat(1e4 / np.maximum(lib, _EPS),
                                           np.diff(block.indptr))),
                 block.indices, block.indptr), shape=block.shape)
            out[start:stop] = (scaled @ weights).toarray()
    out = np.zeros(n_axes)
    for j in range(n_axes):
        x, y = a[:, j], b[:, j]
        out[j] = 0.0 if x.std() <= 0 or y.std() <= 0 else float(np.corrcoef(x, y)[0, 1])
    return np.clip(out, 0.0, 1.0)

def admissible(counts, space: AxisSpace, floor: float | None, seed: int = 42, log=None):
    rel = reliability(counts, space.weights, seed=seed)
    if floor is None:
        keep = np.ones(len(space), dtype=bool)
    else:
        keep = rel >= float(floor)
        if not keep.any():
            raise ValueError(
                f"no axis reaches the split-half reliability floor of {floor}; "
                f"the best is {rel.max():.3f} over {len(space)} axes"
            )
    if log is not None:
        log.record("axis_reliability_floor", None if floor is None else float(floor),
                   section="axes")
        log.record("axis_reliability",
                   {n: round(float(r), 4) for n, r in zip(space.names, rel)},
                   section="axes")
        log.record("n_axes_admissible", int(keep.sum()), section="axes")
        if not keep.all():
            log.record("axes_refused", [n for n, k in zip(space.names, keep) if not k],
                       section="axes")
    return keep, rel

def restrict(space: AxisSpace, keep) -> AxisSpace:
    keep = np.asarray(keep, dtype=bool)
    return AxisSpace(weights=sparse.csr_matrix(space.weights)[:, keep],
                     names=[n for n, k in zip(space.names, keep) if k],
                     mean=space.mean[keep], sd=space.sd[keep],
                     cohort_raw=None if space.cohort_raw is None
                     else space.cohort_raw[:, keep])
