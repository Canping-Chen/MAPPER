from __future__ import annotations

import numpy as np
import torch

def cox_efron_nll(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    nll, n_ev = _cox_efron_terms(risk.reshape(-1), time.reshape(-1),
                                 event.reshape(-1), eps)
    return nll / max(n_ev, 1.0)

def _cox_efron_terms(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float,
) -> "tuple[torch.Tensor, float]":
    order = torch.argsort(time)
    r = risk[order]
    t = time[order]
    ev = event[order]

    shift = r.max().detach()
    exp_r = torch.exp(r - shift)

    rev_cum = torch.flip(torch.cumsum(torch.flip(exp_r, [0]), 0), [0])

    _, inv = torch.unique_consecutive(t, return_inverse=True)
    n_blocks = int(inv[-1].item()) + 1

    evf = ev.to(r.dtype)
    d = torch.zeros(n_blocks, dtype=r.dtype, device=r.device).index_add_(0, inv, evf)
    s_tied = torch.zeros(n_blocks, dtype=r.dtype, device=r.device).index_add_(
        0, inv, exp_r * evf)

    idx_ev = torch.nonzero(ev > 0.5, as_tuple=False).reshape(-1)
    if idx_ev.numel() == 0:

        return (risk * 0.0).sum(), 0.0

    blk = inv[idx_ev]

    first_of_block = torch.searchsorted(blk, blk)
    k = (torch.arange(idx_ev.numel(), device=r.device, dtype=r.dtype)
         - first_of_block.to(r.dtype))

    s_risk = rev_cum[torch.searchsorted(inv, blk)]

    denom = s_risk - (k / d[blk]) * s_tied[blk]

    nll = (torch.log(denom + eps) - (r[idx_ev] - shift)).sum()
    return nll, float(idx_ev.numel())

def fit_ridge_cox(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    lam: float,
    *,
    max_iter: int = 500,
    history_size: int = 50,
    tolerance_grad: float = 1e-7,
):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device="cpu")
    tt = torch.as_tensor(np.asarray(time), dtype=torch.float64, device="cpu")
    et = torch.as_tensor(np.asarray(event), dtype=torch.float64, device="cpu")
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, device="cpu",
                    requires_grad=True)

    opt = torch.optim.LBFGS(
        [w],
        max_iter=int(max_iter),
        history_size=int(history_size),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = cox_efron_nll(Xt @ w, tt, et) + 0.5 * float(lam) * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.enable_grad():
        final = closure()
    info = {
        "lam": float(lam),
        "loss": float(final.detach()),
        "grad_norm": float(w.grad.detach().norm()),
        "n_iter": int(opt.state[opt._params[0]].get("n_iter", 0)),
    }
    return w.detach().numpy().copy(), info

def _balanced_weights(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    pos = label >= 0.5
    n_pos, n_neg = float(pos.sum()), float((~pos).sum())
    w = np.ones(label.shape[0], dtype=np.float64)
    if n_pos > 0 and n_neg > 0:
        w[pos] = 0.5 * label.shape[0] / n_pos
        w[~pos] = 0.5 * label.shape[0] / n_neg
    return w

def fit_ridge_logistic(
    X: np.ndarray,
    label: np.ndarray,
    lam: float,
    *,
    max_iter: int = 500,
    history_size: int = 50,
    tolerance_grad: float = 1e-7,
):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device="cpu")
    yt = torch.as_tensor(np.asarray(label, dtype=np.float64).reshape(-1),
                         dtype=torch.float64, device="cpu")
    st = torch.as_tensor(_balanced_weights(label), dtype=torch.float64,
                         device="cpu")
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, device="cpu",
                    requires_grad=True)

    opt = torch.optim.LBFGS(
        [w], max_iter=int(max_iter), history_size=int(history_size),
        tolerance_grad=float(tolerance_grad), tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        z = Xt @ w
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            z, yt, weight=st, reduction="sum") / st.sum()
        loss = bce + 0.5 * float(lam) * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.enable_grad():
        final = closure()
    info = {"lam": float(lam), "loss": float(final.detach()),
            "grad_norm": float(w.grad.detach().norm()),
            "n_iter": int(opt.state[opt._params[0]].get("n_iter", 0))}
    return w.detach().numpy().copy(), info

def fit_ridge_regression(X: np.ndarray, y: np.ndarray, lam: float):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yc = y - y.mean()
    n, p = X.shape
    lam = float(lam)
    if n <= p:
        K = X @ X.T
        K[np.diag_indices_from(K)] += lam
        alpha = np.linalg.solve(K, yc)
        w = X.T @ alpha
    else:
        A = X.T @ X
        A[np.diag_indices_from(A)] += lam
        w = np.linalg.solve(A, X.T @ yc)
    resid = float(np.mean((X @ w - yc) ** 2))
    return w, {"lam": lam, "loss": resid + 0.5 * lam * float(w @ w),
               "grad_norm": 0.0, "n_iter": 0}

PHENOTYPES = ("survival", "classification", "continuous")

METRIC_NAMES = {"survival": "C-index", "classification": "AUROC",
                "continuous": "Pearson"}

def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1) >= 0.5
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    s_sorted = score[order]
    i = 0
    while i < score.size:
        j = i
        while j + 1 < score.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))

def _harrell_cindex(time: np.ndarray, risk: np.ndarray,
                    event: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    event = np.asarray(event, dtype=np.float64).reshape(-1)
    ev = event > 0.5

    cmp_ = ev[:, None] & (time[None, :] > time[:, None])
    den = float(cmp_.sum())
    if den == 0.0:
        return 0.5
    higher = (risk[:, None] > risk[None, :]) & cmp_
    tied = (risk[:, None] == risk[None, :]) & cmp_
    return float((higher.sum() + 0.5 * tied.sum()) / den)

def _oof_metric(scores: np.ndarray, y: np.ndarray, phenotype: str) -> float:
    if phenotype == "survival":
        return _harrell_cindex(y[:, 0], scores, y[:, 1])
    if phenotype == "classification":
        return _auroc(scores, y)
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / den) if den > 0 else 0.0
