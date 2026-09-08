from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

class StratifiedAUROC(float):

    n_strata_used: int

    def __new__(cls, value: float, n_strata_used: int) -> "StratifiedAUROC":
        obj = super().__new__(cls, value)
        obj.n_strata_used = int(n_strata_used)
        return obj

    def __repr__(self) -> str:
        return f"StratifiedAUROC({float(self)!r}, n_strata_used={self.n_strata_used})"

def _as_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype == bool:
        return labels.ravel()
    values = labels.ravel().astype(float)
    bad = ~np.isin(values, (0.0, 1.0))
    if bad.any():
        raise ValueError("labels must be boolean or 0/1; got other values")
    return values > 0.5

def auroc(score: np.ndarray, labels: np.ndarray) -> float:
    score = np.asarray(score, dtype=float).ravel()
    labels = _as_labels(labels)
    if score.size != labels.size:
        raise ValueError(f"score has {score.size} units, labels has {labels.size}")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def auroc_distance(value: float) -> float:
    return float(abs(float(value) - 0.5))

def enrichment(observed: float, base_rate: float) -> float:
    base_rate = float(base_rate)
    if not np.isfinite(base_rate) or base_rate <= 0.0:
        return float("nan")
    return float(observed) / base_rate

def depth_strata(library: np.ndarray, n_strata: int = 10) -> np.ndarray:
    library = np.asarray(library, dtype=float).ravel()
    if n_strata < 1:
        raise ValueError(f"n_strata must be at least 1, got {n_strata}")
    if library.size == 0:
        return np.zeros(0, dtype=int)
    if not np.isfinite(library).all():
        raise ValueError("library contains non-finite values")
    edges = np.quantile(library, np.linspace(0.0, 1.0, n_strata + 1))
    inner = np.unique(edges[1:-1])
    inner = inner[(inner > library.min()) & (inner <= library.max())]
    return np.searchsorted(inner, library, side="left").astype(int)

def stratified_auroc(
    score: np.ndarray,
    labels: np.ndarray,
    stratum: np.ndarray,
    min_per_stratum: int = 20,
) -> float:
    score = np.asarray(score, dtype=float).ravel()
    labels = _as_labels(labels)
    stratum = np.asarray(stratum).ravel()
    if not (score.size == labels.size == stratum.size):
        raise ValueError(
            f"score/labels/stratum lengths differ: {score.size}, "
            f"{labels.size}, {stratum.size}"
        )
    total_weight = 0.0
    total = 0.0
    used = 0
    for value in np.unique(stratum):
        take = stratum == value
        y = labels[take]
        n_pos = int(y.sum())
        n_neg = int(y.size - n_pos)
        if n_pos < min_per_stratum or n_neg < min_per_stratum:
            continue
        inner = auroc(score[take], y)
        if not np.isfinite(inner):
            continue
        weight = float(n_pos) * float(n_neg)
        total += weight * inner
        total_weight += weight
        used += 1
    if used == 0 or total_weight <= 0.0:
        return StratifiedAUROC(float("nan"), 0)
    return StratifiedAUROC(total / total_weight, used)

def depth_report(
    score: np.ndarray,
    labels: np.ndarray,
    library: np.ndarray,
    n_detected: np.ndarray | None,
    n_strata: int = 10,
) -> dict:
    labels = _as_labels(labels)
    library = np.asarray(library, dtype=float).ravel()
    strata = depth_strata(library, n_strata=n_strata)
    strat = stratified_auroc(score, labels, strata)

    value = auroc(score, labels)
    depth = auroc(library, labels)
    detected = (
        float("nan") if n_detected is None else auroc(n_detected, labels)
    )
    return {
        "auroc": float(value),
        "depth_auroc": float(depth),
        "n_detected_auroc": float(detected),
        "stratified_auroc": float(strat),
        "n_strata_used": int(strat.n_strata_used),
        "n_strata_available": int(np.unique(strata).size),
        "base_rate": float(labels.mean()) if labels.size else float("nan"),
        "n_units": int(labels.size),
        "n_positive": int(labels.sum()),
        "auroc_distance": auroc_distance(value),
        "depth_auroc_distance": auroc_distance(depth),
        "n_detected_auroc_distance": auroc_distance(detected),
        "stratified_auroc_distance": auroc_distance(strat),
    }
