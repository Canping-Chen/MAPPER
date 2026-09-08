from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

INTEGER_RTOL = 1e-5
INTEGER_ATOL = 1e-3

@dataclass(frozen=True)
class SlideMatrix:

    counts: sparse.csr_matrix
    genes: np.ndarray
    library: np.ndarray
    semantics: str
    is_counts: bool

    sequencing_library: np.ndarray | None = None

def _as_csr(X) -> sparse.csr_matrix:
    if X is None:
        raise ValueError("adata.X is None; there is no matrix to model")
    if sparse.issparse(X):
        return sparse.csr_matrix(X, dtype=np.float64)
    return sparse.csr_matrix(np.asarray(X, dtype=np.float64))

def _same_sparsity(m: sparse.csr_matrix, data: np.ndarray) -> sparse.csr_matrix:
    out = sparse.csr_matrix((data, m.indices.copy(), m.indptr.copy()), shape=m.shape)
    out.eliminate_zeros()
    return out

def _is_integer(values: np.ndarray) -> bool:
    return bool(np.all(values == np.rint(values)))

def _is_integer_after_expm1(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    linear = np.expm1(values.astype(np.float64))
    deviation = np.abs(linear - np.rint(linear))
    allowed = np.maximum(INTEGER_ATOL, INTEGER_RTOL * np.abs(linear))
    return bool(np.all(deviation <= allowed))

def _read_counts_layer(adata, name, log):
    layers = getattr(adata, "layers", {})
    if name not in layers:
        raise KeyError(
            f"counts_layer={name!r} is not in layers; available layers are "
            f"{sorted(layers.keys())}"
        )
    matrix = _as_csr(layers[name])
    if matrix.data.size and not _is_integer(matrix.data):
        raise ValueError(
            f"layer {name!r} is not integer-valued, so it is not counts; "
            "point counts_layer at the raw layer or leave it unset"
        )
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError(f"layer {name!r} has negative values")
    if log is not None:
        log.record("counts_layer", name)
        log.record("X_semantics", f"counts_layer:{name}")
    library = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return SlideMatrix(counts=matrix, genes=np.asarray(adata.var_names, dtype=object),
                       library=library, semantics=f"counts_layer:{name}",
                       is_counts=True)

SEQ_DEPTH_COLUMNS = ("nCount_Spatial", "nCount_RNA")

def _sequencing_library(adata, n_units: int):
    obs = getattr(adata, "obs", None)
    if obs is None:
        return None, None
    for name in SEQ_DEPTH_COLUMNS:
        if name not in obs.columns:
            continue
        v = np.asarray(obs[name], dtype=float).ravel()
        if v.size != n_units:
            raise ValueError(
                f"obs[{name!r}] has {v.size} entries for {n_units} units"
            )
        if not np.isfinite(v).all():
            raise ValueError(
                f"obs[{name!r}] holds non-finite values; it is meant to be the "
                f"pre-correction sequencing depth and cannot be guessed at"
            )
        return v, f"obs:{name}"
    return None, None

def read_slide_matrix(
    adata,
    *,
    allow_normalized_surrogate: bool = False,
    counts_layer: str | None = None,
    log=None,
) -> SlideMatrix:
    if counts_layer is not None:
        return _read_counts_layer(adata, counts_layer, log)
    matrix = _as_csr(adata.X)
    genes = np.asarray(adata.var_names, dtype=object)
    if genes.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"X has {matrix.shape[1]} columns but var_names has {genes.shape[0]}"
        )
    data = matrix.data

    if data.size and not np.isfinite(data).all():
        raise ValueError("X holds non-finite values; refusing to guess what they meant")
    if data.size and data.min() < 0:
        raise ValueError(
            f"X holds negative values (min {data.min():.4g}): this is scaled or "
            f"residual data, and compositional information is destroyed by it. "
            f"Supply a matrix on a non-negative scale."
        )

    altered_upstream = False
    if _is_integer(data):
        semantics, counts, is_counts = "counts", matrix, True
        library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    elif _is_integer_after_expm1(data):
        semantics, is_counts, altered_upstream = "log1p_integer_inverted", True, True
        counts = _same_sparsity(matrix, np.rint(np.expm1(data.astype(np.float64))))
        library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    else:
        semantics, is_counts = "normalized_surrogate", False
        if not allow_normalized_surrogate:
            raise ValueError(
                "X is a normalized_surrogate: non-negative floats that are "
                "neither counts nor log1p of counts, so per-unit sequencing "
                "depth is not recoverable and a count likelihood would be a "
                "fiction. Pass allow_normalized_surrogate=True to proceed on a "
                "declared, recorded fallback."
            )
        if log is None:
            raise ValueError(
                "allow_normalized_surrogate=True requires a log: the fallback "
                "must be recorded, not taken silently"
            )
        log.degrade(
            "X_semantics",
            reason=(
                "X is neither counts nor log1p of counts, so per-unit depth is "
                "not recoverable from it"
            ),
            fallback=(
                "treat expm1(X) as a relative profile and give every unit the "
                "same constant effective library (the median of expm1(X).sum(1))"
            ),
        )
        counts = _same_sparsity(matrix, np.expm1(data.astype(np.float64)))
        totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        library = np.full(matrix.shape[0], float(np.median(totals)))

    seq_library, seq_source = ((None, None) if not altered_upstream
                               else _sequencing_library(adata, matrix.shape[0]))

    if log is not None:
        log.record("X_semantics", semantics)
        log.record("is_counts", bool(is_counts), section="slide")
        log.record("library_altered_upstream", bool(altered_upstream), section="slide")
        log.record("n_units", int(matrix.shape[0]), section="slide")
        log.record("n_genes", int(genes.shape[0]), section="slide")
        log.record("library_median", float(np.median(library)), section="slide")
        log.record("library_min", float(library.min()) if library.size else 0.0,
                   section="slide")
        log.record("library_max", float(library.max()) if library.size else 0.0,
                   section="slide")
        if altered_upstream:
            log.record("sequencing_library_source", seq_source, section="slide")
            if seq_library is not None:
                log.record("sequencing_library_median",
                           float(np.median(seq_library)), section="slide")
                log.record("sequencing_library_min", float(seq_library.min()),
                           section="slide")
                log.record("sequencing_library_max", float(seq_library.max()),
                           section="slide")
            log.record(
                "library_note",
                "X was log1p of upstream-corrected counts; the per-unit depth "
                "reported here is the corrected depth, not sequencing depth",
                section="slide",
            )

    return SlideMatrix(counts=counts, genes=genes, library=library,
                       semantics=semantics, is_counts=is_counts,
                       sequencing_library=seq_library)

def detection_fraction(counts) -> np.ndarray:
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(np.asarray(counts, dtype=float))
    matrix = counts.tocsr(copy=False)
    n_units = matrix.shape[0]
    if n_units == 0:
        raise ValueError("cannot compute detection fractions from zero units")

    n_genes = matrix.shape[1]
    idx, data = matrix.indices, matrix.data
    has_explicit_zeros = bool(data.size) and np.count_nonzero(data) != data.size
    counts_per_gene = np.zeros(n_genes, dtype=np.int64)
    step = 2_000_000
    for start in range(0, idx.size, step):
        chunk = idx[start:start + step]
        if has_explicit_zeros:
            chunk = chunk[data[start:start + step] != 0]
        counts_per_gene += np.bincount(chunk, minlength=n_genes)
    return counts_per_gene.astype(float) / float(n_units)
