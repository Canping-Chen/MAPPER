from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import (_harrell_cindex, fit_ridge_cox, fit_ridge_logistic,
                         fit_ridge_regression)

EVENTS_PER_PARAMETER = 10.0

MIN_LAMBDA = 4.0

ENDPOINTS = ("survival", "classification", "continuous")

@dataclass(frozen=True)
class Direction:
    beta: np.ndarray
    mean: float
    sd: float
    lam: float
    effective_df: float
    n_events: float
    c_index: float
    c_index_ci: tuple
    passed: bool

    gamma: np.ndarray | None = None

    anchor_hi: float = 0.0
    anchor_lo: float = 0.0

    def index(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (x @ self.beta - self.mean) / self.sd

def effective_df(design: np.ndarray, lam: float) -> float:
    d2 = np.linalg.svd(np.asarray(design, float), compute_uv=False) ** 2
    return float(np.sum(d2 / (d2 + lam))) if lam > 0 else float(np.sum(d2 > 0))

def lambda_for_df(design: np.ndarray, target: float) -> float:
    d2 = np.linalg.svd(np.asarray(design, float), compute_uv=False) ** 2
    if target >= float(np.sum(d2 > 0)):
        return 0.0
    lo, hi = 0.0, max(float(d2.max()), 1.0)
    while float(np.sum(d2 / (d2 + hi))) > target and hi < 1e18:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.sum(d2 / (d2 + mid))) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def reference_groups(clinical, endpoint: str):
    if endpoint == "survival":
        t = np.asarray(clinical["time"], float)
        e = np.asarray(clinical["event"], float)
        land = float(np.nanquantile(t, 0.5))
        return (e >= 0.5) & (t <= land), (e < 0.5) & (t > land)
    if endpoint == "classification":
        y = np.asarray(clinical["label"]).astype(bool)
        return y, ~y
    v = np.asarray(clinical["value"], float)
    return v >= np.quantile(v, 0.75), v <= np.quantile(v, 0.25)

def _n_events(clinical, endpoint: str) -> float:
    if endpoint == "survival":
        return float(np.asarray(clinical["event"], float).sum())
    if endpoint == "classification":
        y = np.asarray(clinical["label"]).astype(bool)
        return float(min(y.sum(), (~y).sum()))
    return float(np.asarray(clinical["value"]).size)

def _fit(x, clinical, endpoint, lam) -> np.ndarray:
    if endpoint == "survival":
        return np.asarray(fit_ridge_cox(x, clinical["time"], clinical["event"], lam)[0])
    if endpoint == "classification":
        return np.asarray(fit_ridge_logistic(x, clinical["label"], lam)[0])
    if endpoint == "continuous":
        return np.asarray(fit_ridge_regression(x, clinical["value"], lam)[0])
    raise ValueError(f"unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")

def _concordance(clinical, endpoint, eta) -> float:
    if endpoint == "survival":
        return float(_harrell_cindex(np.asarray(clinical["time"], float), eta,
                                     np.asarray(clinical["event"], float)))
    if endpoint == "classification":
        from .objectives import _auroc
        return float(_auroc(eta, np.asarray(clinical["label"]).astype(float)))
    v = np.asarray(clinical["value"], float)
    return float(0.5 + 0.5 * np.corrcoef(eta, v)[0, 1]) if v.std() > 0 else 0.5

def _crossfit(w, clinical, endpoint, lam, seed, folds=5, n_bag=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = rng.permutation(w.shape[0]) % folds
    eta = np.empty(w.shape[0])
    for f in range(folds):
        tr, te = fold != f, fold == f
        sub = {k: np.asarray(v)[tr] for k, v in clinical.items()}
        eta[te] = w[te] @ _bagged_beta(w[tr], sub, endpoint, lam,
                                       max(1, min(n_bag, 12)), seed + f)
    return eta

def _row_space(w: np.ndarray):
    centre = w.mean(axis=0)
    u, d, vt = np.linalg.svd(w - centre, full_matrices=False)
    r = int((d > d.max() * 1e-10).sum()) if d.size else 0
    return centre, vt[:r].T, d[:r]

@dataclass(frozen=True)
class DesignBasis:

    centre: np.ndarray | None
    basis: np.ndarray | None
    design: np.ndarray
    lam: float

    def to_features(self, gamma: np.ndarray) -> np.ndarray:
        return self.basis @ gamma if self.basis is not None else gamma

def design_basis(w, clinical, endpoint: str) -> DesignBasis:
    w = np.asarray(w, dtype=float)
    n_events = _n_events(clinical, endpoint)
    if w.shape[1] > w.shape[0]:
        centre, basis, _ = _row_space(w)
        design = (w - centre) @ basis
    else:
        centre, basis, design = None, None, w
    lam = max(MIN_LAMBDA, lambda_for_df(design, max(n_events / EVENTS_PER_PARAMETER, 1.0)))
    return DesignBasis(centre=centre, basis=basis, design=design, lam=lam)

def _bagged_beta(w, clinical, endpoint, lam, n_bag, seed) -> np.ndarray:
    if n_bag <= 1:
        return _fit(w, clinical, endpoint, lam)
    n = w.shape[0]
    rng = np.random.default_rng(seed)
    wide = w.shape[1] > n
    if wide:
        centre, basis, _ = _row_space(w)
        design = (w - centre) @ basis
    else:
        design = w
    acc, used = np.zeros(design.shape[1]), 0
    for _ in range(int(n_bag)):
        idx = rng.integers(0, n, n)
        sub = {k: np.asarray(v)[idx] for k, v in clinical.items()}
        if endpoint == "survival" and float(np.asarray(sub["event"]).sum()) < 2:
            continue
        try:
            acc += _fit(design[idx], sub, endpoint, lam)
            used += 1
        except Exception:
            continue
    if used == 0:
        return _fit(w, clinical, endpoint, lam)
    beta = acc / used
    return basis @ beta if wide else beta

def fit_direction(w, clinical, endpoint: str, seed: int = 42, n_boot: int = 200,
                  n_bag: int = 1, basis=None, log=None) -> Direction:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")
    w = np.asarray(w, dtype=float)
    n_events = _n_events(clinical, endpoint)
    b = basis if basis is not None else design_basis(w, clinical, endpoint)
    lam = b.lam
    gamma = _bagged_beta(b.design, clinical, endpoint, lam, n_bag, seed)
    beta = b.to_features(gamma)
    raw = w @ beta
    sd = float(raw.std()) or 1.0

    if int(n_boot) <= 0:
        bad, _g = reference_groups(clinical, endpoint)
        idx0 = (raw - float(raw.mean())) / sd
        h = float(np.median(idx0))
        return Direction(beta=beta, mean=float(raw.mean()), sd=sd, lam=float(lam),
                         effective_df=float("nan"), n_events=n_events,
                         c_index=float("nan"),
                         c_index_ci=(float("nan"), float("nan")), passed=False,
                         gamma=gamma, anchor_hi=h,
                         anchor_lo=h - (float(np.median(idx0[bad])) - h) if bad.any() else h)

    eta = _crossfit(b.design, clinical, endpoint, lam, seed, n_bag=n_bag)
    c_index = _concordance(clinical, endpoint, eta)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(eta), len(eta))
        sub = {k: np.asarray(v)[idx] for k, v in clinical.items()}
        try:
            boot.append(_concordance(sub, endpoint, eta[idx]))
        except Exception:
            continue
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
        if len(boot) >= 20 else (float("nan"), float("nan"))
    passed = bool(np.isfinite(lo) and lo > 0.5)

    bad, _good = reference_groups(clinical, endpoint)
    idx = (raw - float(raw.mean())) / sd
    a_hi = float(np.median(idx))
    a_lo = a_hi - (float(np.median(idx[bad])) - a_hi) if bad.any() else a_hi
    d = Direction(beta=beta, mean=float(raw.mean()), sd=sd, lam=float(lam),
                  effective_df=effective_df(b.design, lam), n_events=n_events,
                  c_index=c_index, c_index_ci=(lo, hi), passed=passed, gamma=gamma,
                  anchor_hi=a_hi, anchor_lo=a_lo)
    if log is not None:
        log.record("endpoint", endpoint, section="head")
        log.record("n_events", n_events, section="head")
        log.record("n_axes", int(w.shape[1]), section="head")
        log.record("ridge_lambda", d.lam, section="head")
        log.record("effective_df", round(d.effective_df, 3), section="head")
        log.record("events_per_effective_parameter",
                   round(n_events / max(d.effective_df, 1e-9), 2), section="head")
        log.record("c_index_crossfit", round(c_index, 4), section="head")
        log.record("c_index_ci", [round(lo, 4), round(hi, 4)], section="head")
        log.record("n_bag", int(n_bag), section="head")
        log.record("anchor_hi", round(a_hi, 4), section="head")
        log.record("anchor_lo", round(a_lo, 4), section="head")
        log.record("direction_gate_passed", passed, section="head")
    return d

def score(direction: Direction, unit: np.ndarray, niche: np.ndarray | None,
          niche_weight: float) -> np.ndarray:
    if niche is None or niche_weight <= 0:
        return direction.index(unit)
    return ((1.0 - niche_weight) * direction.index(unit)
            + niche_weight * direction.index(niche))
