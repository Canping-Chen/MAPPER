from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objectives import _auroc, _harrell_cindex

ENDPOINTS = ("survival", "classification", "continuous")

SCORE_CHUNK = 65_536

EVENTS_PER_PARAMETER = 10.0
MIN_LAMBDA = 4.0

MLP_HIDDEN = 64
MLP_DROPOUT = 0.3
MLP_WEIGHT_DECAY = 1e-2
MLP_LR = 5e-3
MLP_MAX_EPOCHS = 400
MLP_PATIENCE = 30
MLP_VAL_FRACTION = 0.2
MLP_REPLICATES = 20

@dataclass(frozen=True)
class CohortSpan:

    centre: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(x, float)) - self.centre) @ self.basis

def cohort_span(w: np.ndarray) -> CohortSpan | None:
    w = np.asarray(w, dtype=float)
    if w.shape[1] <= w.shape[0]:
        return None
    centre = w.mean(axis=0)
    _, d, vt = np.linalg.svd(w - centre, full_matrices=False)
    r = int((d > d.max() * 1e-10).sum()) if d.size else 0
    if r == 0:
        raise ValueError("the cohort's profiles span no directions at all")
    return CohortSpan(centre=centre, basis=vt[:r].T, singular_values=d[:r])

@dataclass(frozen=True)
class Teacher:
    kind: str
    endpoint: str
    predict: object = field(repr=False)
    mean: float = 0.0
    sd: float = 1.0
    detail: dict = field(default_factory=dict)
    span: CohortSpan | None = None

    def score(self, x: np.ndarray, chunk: int = SCORE_CHUNK) -> np.ndarray:
        x = np.atleast_2d(x)
        n = x.shape[0]
        out = np.empty(n, dtype=float)
        for start in range(0, n, int(chunk)):
            stop = min(start + int(chunk), n)
            block = np.asarray(x[start:stop], dtype=float)
            if self.span is not None:
                block = self.span(block)
            out[start:stop] = np.asarray(self.predict(block), float).ravel()
        return (out - self.mean) / self.sd

def _fit_linear(w, clinical, endpoint, log=None):
    from .head import (_fit, _n_events, design_basis, effective_df)

    b = design_basis(w, clinical, endpoint)
    gamma = _fit(b.design, clinical, endpoint, b.lam)
    beta = b.to_features(gamma)
    if log is not None:
        log.record("teacher", "linear", section="teacher")
        log.record("ridge_lambda", round(float(b.lam), 3), section="teacher")
        log.record("effective_df", round(effective_df(b.design, b.lam), 2),
                   section="teacher")
        log.record("events_per_effective_parameter",
                   round(_n_events(clinical, endpoint)
                         / max(effective_df(b.design, b.lam), 1e-9), 2),
                   section="teacher")
    return lambda x: np.asarray(x, float) @ beta, {"beta": beta}

def _fit_mlp(w, clinical, endpoint, seed, log=None):
    import torch
    import torch.nn as nn

    from .objectives import cox_efron_nll

    w = np.asarray(w, dtype=np.float64)
    n, p = w.shape
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    X = torch.as_tensor(w, dtype=torch.float64)

    def loss_for(idx):
        if endpoint == "survival":
            t = torch.as_tensor(np.asarray(clinical["time"], float)[idx])
            e = torch.as_tensor(np.asarray(clinical["event"], float)[idx])
            return lambda s: cox_efron_nll(s, t, e)
        if endpoint == "classification":
            y = torch.as_tensor(np.asarray(clinical["label"]).astype(np.float64)[idx])
            pos = float(y.sum()) or 1.0
            wt = torch.where(y > 0, len(y) / (2 * pos),
                             len(y) / (2 * max(len(y) - pos, 1.0)))
            return lambda s: nn.functional.binary_cross_entropy_with_logits(
                s, y, weight=wt)
        v = torch.as_tensor(np.asarray(clinical["value"], float)[idx])
        v = v - v.mean()
        return lambda s: ((s - v) ** 2).mean()

    nets, used = [], 0
    for rep in range(MLP_REPLICATES):

        boot = rng.integers(0, n, n)
        tr = boot
        va = np.setdiff1d(np.arange(n), np.unique(boot))
        if va.size < 5:
            continue
        if endpoint == "survival" and (float(np.asarray(clinical["event"])[tr].sum()) < 3
                                       or float(np.asarray(clinical["event"])[va].sum()) < 2):
            continue
        net = nn.Sequential(nn.Linear(p, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
                            nn.GELU(), nn.Dropout(MLP_DROPOUT),
                            nn.Linear(MLP_HIDDEN, 1)).to(torch.float64)
        opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR,
                                weight_decay=MLP_WEIGHT_DECAY)
        tr_loss, va_loss = loss_for(tr), loss_for(va)
        best, best_state, wait = np.inf, None, 0
        for _ in range(MLP_MAX_EPOCHS):
            net.train(); opt.zero_grad()
            tr_loss(net(X[tr]).reshape(-1)).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float(va_loss(net(X[va]).reshape(-1)))
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= MLP_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        nets.append(net); used += 1

    if not nets:
        raise ValueError("no MLP replicate could be trained; the cohort is too small")

    def predict(x):
        t = torch.as_tensor(np.asarray(x, dtype=np.float64))
        with torch.no_grad():
            return np.mean([net(t).reshape(-1).numpy() for net in nets], axis=0)

    if log is not None:
        log.record("teacher", "mlp", section="teacher")
        log.record("hidden", MLP_HIDDEN, section="teacher")
        log.record("dropout", MLP_DROPOUT, section="teacher")
        log.record("weight_decay", MLP_WEIGHT_DECAY, section="teacher")
        log.record("replicates", used, section="teacher")
        log.record("validation", "out-of-bag", section="teacher")
        log.record("n_parameters", int(sum(q.numel() for q in nets[0].parameters())),
                   section="teacher")
    return predict, {"n_replicates": used, "validation": "out-of-bag"}

def _raw_out_of_fold(w, clinical, endpoint, kind, seed, folds):
    n = w.shape[0]
    folds = int(max(2, min(folds, n)))
    fold = np.random.default_rng(seed).permutation(n) % folds
    out = np.full(n, np.nan)
    for f in range(folds):
        test = fold == f
        train = ~test
        sub = {k: np.asarray(v)[train] for k, v in clinical.items()}
        if endpoint == "survival" and float(np.asarray(sub["event"]).sum()) < 3:
            continue
        span = cohort_span(w[train]) if kind == "mlp" else None
        design = span(w[train]) if span is not None else w[train]
        fit = _fit_linear if kind == "linear" else _fit_mlp
        predict = (fit(design, sub, endpoint)[0] if kind == "linear"
                   else fit(design, sub, endpoint, seed)[0])
        held = span(w[test]) if span is not None else w[test]
        out[test] = np.asarray(predict(held), float).ravel()
    if not np.isfinite(out).any():
        raise ValueError("no fold could be fitted; the cohort is too small")
    return out

def fit_teacher(w, clinical, endpoint: str, kind: str = "mlp", seed: int = 42,
                folds: int = 5, log=None) -> Teacher:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")
    w = np.asarray(w, dtype=float)
    span = cohort_span(w) if kind == "mlp" else None
    design = span(w) if span is not None else w
    if kind == "linear":
        predict, detail = _fit_linear(design, clinical, endpoint, log=log)
    elif kind == "mlp":
        predict, detail = _fit_mlp(design, clinical, endpoint, seed, log=log)
    else:
        raise ValueError(f"unknown teacher {kind!r}; expected 'linear' or 'mlp'")

    raw_oof = _raw_out_of_fold(w, clinical, endpoint, kind, seed, folds)
    seen = np.isfinite(raw_oof)
    mean = float(raw_oof[seen].mean())
    sd = float(raw_oof[seen].std()) or 1.0

    detail = dict(detail)
    detail["n_inputs"] = int(design.shape[1])
    detail["basis_singular_values"] = (np.empty(0) if span is None
                                       else span.singular_values)
    detail["calibration"] = "out-of-fold"
    detail["folds"] = int(folds)
    detail["oof"] = (raw_oof - mean) / sd
    t = Teacher(kind=kind, endpoint=endpoint, predict=predict,
                mean=mean, sd=sd, detail=detail, span=span)
    if log is not None:
        log.record("teacher_inputs", int(design.shape[1]), section="teacher")
        log.record("cohort_span_rank", 0 if span is None else int(span.basis.shape[1]),
                   section="teacher")
        log.record("calibration", "out-of-fold", section="teacher")
        log.record("calibration_folds", int(folds), section="teacher")
        log.record("cohort_concordance_insample",
                   round(concordance(t, w, clinical, endpoint), 4), section="teacher")
        log.record("cohort_concordance_crossfit",
                   round(_concordance_of(detail["oof"], clinical, endpoint), 4),
                   section="teacher")
    return t

def _concordance_of(s, clinical, endpoint: str) -> float:
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    if endpoint == "survival":
        return float(_harrell_cindex(np.asarray(clinical["time"], float)[ok], s[ok],
                                     np.asarray(clinical["event"], float)[ok]))
    if endpoint == "classification":
        return float(_auroc(s[ok], np.asarray(clinical["label"]).astype(float)[ok]))
    v = np.asarray(clinical["value"], float)[ok]
    return float(0.5 + 0.5 * np.corrcoef(s[ok], v)[0, 1]) if v.std() > 0 else 0.5

def concordance(teacher: Teacher, w, clinical, endpoint: str) -> float:
    s = teacher.score(w)
    if endpoint == "survival":
        return float(_harrell_cindex(np.asarray(clinical["time"], float), s,
                                     np.asarray(clinical["event"], float)))
    if endpoint == "classification":
        return float(_auroc(s, np.asarray(clinical["label"]).astype(float)))
    v = np.asarray(clinical["value"], float)
    return float(0.5 + 0.5 * np.corrcoef(s, v)[0, 1]) if v.std() > 0 else 0.5
