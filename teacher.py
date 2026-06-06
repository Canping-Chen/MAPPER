"""
MAPPER bulk teacher MLP.

Implements three phenotype heads:
  - "classification": BCE (binary) / CE (multiclass)
  - "survival"      : Cox partial likelihood (Efron tie handling)
  - "continuous"    : MSE + 0.1 * (1 - Pearson)

Once trained, the teacher is FROZEN. The transfer step does NOT use the
teacher's prediction directly on single cells (that suffers from bulk vs
single-cell domain shift).  Instead we extract a **gene-level signature**
via input * gradient and the transfer module computes a rank-based score on
target cells (Plan B in the design doc).
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ._internal_config import TeacherConfig


# ---------------------------------------------------------------------------
# MLP module
# ---------------------------------------------------------------------------
class TeacherMLP(nn.Module):
    """256 -> 64 -> out  (LayerNorm + GELU + Dropout)."""

    def __init__(
        self,
        n_genes: int,
        out_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.out_dim = int(out_dim)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        out = self.head(z)
        return out, z

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def _cox_partial_likelihood_efron(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps: float = 1e-7,
) -> torch.Tensor:
    """Cox partial likelihood with Efron tie handling.

    Parameters
    ----------
    risk  : (N,) predicted risk score (log hazard).
    time  : (N,) survival time.
    event : (N,) {0, 1} event indicator (1 = event observed).
    """
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)

    # Sort by time descending so cumulative sum gives risk-set sums
    order = torch.argsort(time, descending=True)
    risk_s = risk[order]
    time_s = time[order]
    event_s = event[order]

    log_risk = risk_s
    risk_exp = torch.exp(log_risk - log_risk.max())  # subtract max for stability
    cum_risk = torch.cumsum(risk_exp, dim=0)

    # For Efron, we need to handle ties at each event time
    unique_times, inverse = torch.unique_consecutive(time_s, return_inverse=True)
    nll = risk.sum() * 0.0
    for ti in range(unique_times.shape[0]):
        mask = (inverse == ti) & (event_s.bool())
        d = int(mask.sum().item())
        if d == 0:
            continue
        # last index of this tie block in sorted-by-time-desc array
        last_idx = int(torch.where(inverse == ti)[0].max().item())
        risk_set_sum = cum_risk[last_idx]
        # tied events sum
        tied_idx = torch.where(mask)[0]
        tied_sum = risk_exp[tied_idx].sum()
        # Efron: for k = 0..d-1, denom = risk_set_sum - k/d * tied_sum
        for k in range(d):
            denom = risk_set_sum - (k / d) * tied_sum
            nll = nll + torch.log(denom + eps) - log_risk[tied_idx[k]]
    return nll / max(int(event.sum().item()), 1)


def _pearson_loss(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = pred - pred.mean()
    t = y - y.mean()
    r = (p * t).sum() / ((p.norm() + eps) * (t.norm() + eps))
    return 1.0 - r


# ---------------------------------------------------------------------------
# teacher trainer
# ---------------------------------------------------------------------------
@dataclass
class TeacherFitResult:
    model: TeacherMLP
    metric_name: str
    val_metric: float
    out_dim: int
    cls_pos_index: Optional[int] = None
    val_history: list = None


def _resolve_phenotype_target(
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str],
    survival_event_col: Optional[str],
    class_col: Optional[str],
    positive_class: Optional[str],
    continuous_col: Optional[str],
) -> Tuple[np.ndarray, str, int, Optional[int]]:
    """Build the supervised target.

    Returns ``(y, mode, out_dim, pos_index)``.

    For classification we use BCE on binary problems; multiclass collapses
    to softmax cross-entropy in which case ``pos_index`` is the integer
    code of ``positive_class``.
    """
    if phenotype_type == "classification":
        if class_col is None:
            raise ValueError("phenotype_type='classification' requires class_col")
        labels = bulk_meta[class_col].astype(str).to_numpy()
        uniq = sorted(np.unique(labels).tolist())
        if positive_class is not None:
            if str(positive_class) not in uniq:
                raise ValueError(
                    f"positive_class={positive_class!r} not found in {class_col} (unique: {uniq})"
                )
            if len(uniq) == 2:
                y = (labels == str(positive_class)).astype(np.float32)
                return y, "binary", 1, 1
            # multi-class: treat as softmax CE; signature taken w.r.t. the positive logit
            mapping = {c: i for i, c in enumerate(uniq)}
            y = np.array([mapping[v] for v in labels], dtype=np.int64)
            pos_idx = int(mapping[str(positive_class)])
            return y, "multiclass", len(uniq), pos_idx
        # no positive_class given -> assume binary with the rarer class as 1
        if len(uniq) != 2:
            raise ValueError(
                f"class_col has {len(uniq)} unique values; please supply positive_class."
            )
        # rarer class as positive
        counts = {c: int((labels == c).sum()) for c in uniq}
        positive = min(counts, key=counts.get)
        y = (labels == positive).astype(np.float32)
        return y, "binary", 1, 1

    if phenotype_type == "survival":
        if survival_time_col is None or survival_event_col is None:
            raise ValueError(
                "phenotype_type='survival' requires survival_time_col + survival_event_col"
            )
        t = bulk_meta[survival_time_col].astype(float).to_numpy()
        e = bulk_meta[survival_event_col].astype(float).to_numpy()
        y = np.stack([t, e], axis=1).astype(np.float32)
        return y, "survival", 1, None

    if phenotype_type == "continuous":
        if continuous_col is None:
            raise ValueError("phenotype_type='continuous' requires continuous_col")
        y = bulk_meta[continuous_col].astype(float).to_numpy().astype(np.float32)
        return y, "continuous", 1, None

    raise ValueError(f"Unknown phenotype_type: {phenotype_type}")


def fit_teacher(
    bulk_X: np.ndarray,                # (n_b, g) aligned to shared genes
    bulk_meta: pd.DataFrame,
    phenotype_type: str,
    *,
    survival_time_col: Optional[str] = None,
    survival_event_col: Optional[str] = None,
    class_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    continuous_col: Optional[str] = None,
    config: Optional[TeacherConfig] = None,
    device: Optional[torch.device] = None,
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> TeacherFitResult:
    cfg = config or TeacherConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = logger or logging.getLogger("MAPPER")

    n_b, n_g = bulk_X.shape
    y, mode, out_dim, pos_idx = _resolve_phenotype_target(
        bulk_meta,
        phenotype_type,
        survival_time_col=survival_time_col,
        survival_event_col=survival_event_col,
        class_col=class_col,
        positive_class=positive_class,
        continuous_col=continuous_col,
    )

    # train / val split
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n_b)
    n_val = max(1, int(n_b * cfg.val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xtr = torch.as_tensor(bulk_X[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.as_tensor(bulk_X[val_idx], dtype=torch.float32, device=device)
    if mode == "multiclass":
        ytr = torch.as_tensor(y[tr_idx], dtype=torch.long, device=device)
        yva = torch.as_tensor(y[val_idx], dtype=torch.long, device=device)
    elif mode == "binary":
        ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)
        yva = torch.as_tensor(y[val_idx], dtype=torch.float32, device=device)
    elif mode == "survival":
        ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)
        yva = torch.as_tensor(y[val_idx], dtype=torch.float32, device=device)
    else:  # continuous
        ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)
        yva = torch.as_tensor(y[val_idx], dtype=torch.float32, device=device)

    model = TeacherMLP(
        n_genes=n_g,
        out_dim=out_dim,
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    bs = min(cfg.batch_size_max, max(8, int(math.ceil(len(tr_idx) / 4))))

    def _loss(out, target, mode):
        if mode == "binary":
            return torch.nn.functional.binary_cross_entropy_with_logits(
                out.view(-1), target.view(-1)
            )
        if mode == "multiclass":
            return torch.nn.functional.cross_entropy(out, target)
        if mode == "survival":
            time = target[:, 0]; event = target[:, 1]
            return _cox_partial_likelihood_efron(out.view(-1), time, event)
        # continuous
        out_v = out.view(-1)
        target_v = target.view(-1)
        mse = torch.nn.functional.mse_loss(out_v, target_v)
        peason = _pearson_loss(out_v, target_v)
        return mse + 0.1 * peason

    def _val_metric(model, mode):
        model.eval()
        with torch.no_grad():
            out, _ = model(Xva)
            if mode == "binary":
                p = torch.sigmoid(out.view(-1))
                return _auroc(p.cpu().numpy(), y[val_idx])
            if mode == "multiclass":
                p = torch.softmax(out, dim=1)
                pred = p.argmax(dim=1).cpu().numpy()
                return float((pred == y[val_idx]).mean())
            if mode == "survival":
                risk = out.view(-1).cpu().numpy()
                t = y[val_idx, 0]; e = y[val_idx, 1]
                return _cindex(t, risk, e)
            # continuous
            pred = out.view(-1).cpu().numpy()
            return _pearson(pred, y[val_idx])

    best = (-math.inf if mode != "survival" else -math.inf)
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = []
    metric_name = {
        "binary": "AUROC", "multiclass": "Acc", "survival": "C-index", "continuous": "Pearson",
    }[mode]

    for ep in range(cfg.max_epochs):
        model.train()
        perm_in = torch.randperm(len(tr_idx), device=device)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, len(tr_idx), bs):
            ix = perm_in[s : s + bs]
            xb = Xtr[ix]
            yb = ytr[ix]
            opt.zero_grad()
            out, _ = model(xb)
            loss = _loss(out, yb, mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach().cpu())
            n_batches += 1
        sched.step()
        m = _val_metric(model, mode)
        history.append({"epoch": ep, "train_loss": ep_loss / max(n_batches, 1), "val": m})
        if m > best + 1e-6:
            best = m
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_sd)
    for p in model.parameters():
        p.requires_grad = False

    log.info(
        "[teacher] mode=%s n_b=%d n_g=%d epochs=%d best %s=%.4f",
        mode, n_b, n_g, len(history), metric_name, best,
    )
    return TeacherFitResult(
        model=model,
        metric_name=metric_name,
        val_metric=float(best),
        out_dim=out_dim,
        cls_pos_index=pos_idx,
        val_history=history,
    )


# ---------------------------------------------------------------------------
# small metric utils (no sklearn dependency for survival / pearson)
# ---------------------------------------------------------------------------
def _auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUROC via sorted-by-score cumulative TP/FP."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(int(y.sum()), 1)
    fpr = fp / max(int((1 - y).sum()), 1)
    auc = float(np.trapezoid(tpr, fpr))
    return auc


def _cindex(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index. O(N^2) — fine for n_b <= a few thousand."""
    time = np.asarray(time)
    risk = np.asarray(risk)
    event = np.asarray(event)
    n = len(time)
    num = 0
    den = 0
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] > time[i]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return float(num / den) if den > 0 else 0.5


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
