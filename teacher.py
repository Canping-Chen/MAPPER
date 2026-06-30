"""
MAPPER bulk teacher MLP.

Implements three phenotype heads:
  - "classification": BCE (binary)
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
    refit_full: bool = False        # whether stage-2 full-data refit was run
    best_epoch: int = -1            # stage-1 epoch (0-based) with best val metric
    final_epochs: int = -1          # epochs the returned model was trained for


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

    For classification we use BCE (binary only).
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
            if len(uniq) != 2:
                raise ValueError(
                    f"phenotype_type='classification' supports binary only "
                    f"({len(uniq)} unique values found in {class_col})"
                )
            y = (labels == str(positive_class)).astype(np.float32)
            return y, "binary", 1, 1
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


def _stratified_val_split(
    y: np.ndarray,
    mode: str,
    val_frac: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train / val index split.

    Strata:
      * ``binary``     -> class label (0/1)
      * ``survival``   -> event indicator (``y[:, 1]``)
      * ``continuous`` -> quartile bins of the target value

    Within each stratum a fraction ``val_frac`` (rounded) goes to validation,
    so the validation set keeps the same class / event / value-range balance as
    the full cohort — important for rare-event survival and imbalanced binary
    cohorts where a plain random split can leave val with too few (or zero)
    positives.  Falls back to a single random holdout if the stratified split
    would leave train or val empty.
    """
    n_b = int(y.shape[0])
    rng = np.random.default_rng(random_seed)

    if mode == "binary":
        strata = np.asarray(y, dtype=np.int64).reshape(-1)
    elif mode == "survival":
        strata = np.asarray(y[:, 1], dtype=np.int64).reshape(-1)
    elif mode == "continuous":
        vals = np.asarray(y, dtype=float).reshape(-1)
        n_bins = 4 if n_b >= 10 else 1
        if n_bins <= 1 or np.unique(vals).size < n_bins:
            strata = np.zeros(n_b, dtype=np.int64)          # degenerate -> random
        else:
            edges = np.quantile(vals, [0.25, 0.5, 0.75])
            strata = np.digitize(vals, edges).astype(np.int64)
    else:
        strata = np.zeros(n_b, dtype=np.int64)

    val_mask = np.zeros(n_b, dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_val_s = int(round(val_frac * idx.size))
        val_mask[idx[:n_val_s]] = True

    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(~val_mask)[0]

    # fallback: never return an empty train or val set
    if val_idx.size == 0 or tr_idx.size == 0:
        perm = rng.permutation(n_b)
        n_val = max(1, min(n_b - 1, int(n_b * val_frac)))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

    return tr_idx, val_idx


def _bce_pos_weight(y_arr: np.ndarray, mode: str, device) -> Optional[torch.Tensor]:
    """BCE ``pos_weight = N_neg / N_pos`` for binary classification.

    Gives the rarer class proportionally more weight so the loss is not
    dominated by the majority class.  Symmetric: when positives are the
    majority the weight is < 1 (down-weights positives, i.e. relatively
    up-weights the rarer negatives).  Returns ``None`` for non-binary modes or
    when either class is absent (nothing to balance).
    """
    if mode != "binary":
        return None
    arr = np.asarray(y_arr).reshape(-1)
    n_pos = float((arr == 1).sum())
    n_neg = float((arr == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return None
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def _minority_frac(y: np.ndarray, mode: str):
    """Minority class / event fraction in [0, 0.5]; ``None`` for continuous.

    binary   : min(p, 1-p) over the 0/1 labels
    survival : min(event_rate, 1-event_rate) over y[:,1]
    continuous: None (no class proportion — only the size criterion applies)
    """
    if mode == "binary":
        r = float(np.mean(np.asarray(y).reshape(-1)))
        return min(r, 1.0 - r)
    if mode == "survival":
        r = float(np.mean(np.asarray(y)[:, 1]))
        return min(r, 1.0 - r)
    return None


def _decide_refit(y: np.ndarray, mode: str, n_b: int, cfg) -> bool:
    """Decide whether to run the stage-2 full-data refit.

    ``cfg.refit_full`` overrides when set (True/False); otherwise AUTO: refit
    when the cohort is small (``n_b < refit_min_bulk``) OR imbalanced (minority
    fraction ``< refit_min_class_frac``).  Boundaries (==threshold) skip refit.
    """
    if cfg.refit_full is not None:
        return bool(cfg.refit_full)
    small = n_b < cfg.refit_min_bulk
    mf = _minority_frac(y, mode)
    imbalanced = (mf is not None) and (mf < cfg.refit_min_class_frac)
    return small or imbalanced


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

    # stratified train / val split (by class / event / value-quartile)
    tr_idx, val_idx = _stratified_val_split(y, mode, cfg.val_frac, random_seed)
    n_val = int(val_idx.size)

    Xtr = torch.as_tensor(bulk_X[tr_idx], dtype=torch.float32, device=device)
    Xva = torch.as_tensor(bulk_X[val_idx], dtype=torch.float32, device=device)
    # Targets are encoded identically across modes; the validation metric reads
    # y[val_idx] directly (numpy), so only the training-side tensor is needed.
    ytr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=device)

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
    pos_weight_tr = _bce_pos_weight(y[tr_idx], mode, device)

    def _loss(out, target, mode, pos_weight=None):
        if mode == "binary":
            return torch.nn.functional.binary_cross_entropy_with_logits(
                out.view(-1), target.view(-1), pos_weight=pos_weight
            )
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
            if mode == "survival":
                risk = out.view(-1).cpu().numpy()
                t = y[val_idx, 0]; e = y[val_idx, 1]
                return _cindex(t, risk, e)
            # continuous
            pred = out.view(-1).cpu().numpy()
            return _pearson(pred, y[val_idx])

    best = (-math.inf if mode != "survival" else -math.inf)
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad_epochs = 0
    history = []
    metric_name = {
        "binary": "AUROC", "survival": "C-index", "continuous": "Pearson",
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
            loss = _loss(out, yb, mode, pos_weight_tr)
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
            best_epoch = ep
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    # ---- Stage 2: optional refit on ALL bulk samples ----
    # Reuse the early-stopping point (best_epoch) found on the holdout, but
    # train a fresh model from scratch on every labelled sample for that many
    # epochs (no val, no early stopping). The LR schedule keeps the same
    # T_max so each epoch's learning rate matches stage 1.
    final_epochs = best_epoch + 1
    refit_done = False
    do_refit = _decide_refit(y, mode, n_b, cfg)
    if do_refit and n_b > n_val:
        torch.manual_seed(random_seed)
        Xall = torch.as_tensor(bulk_X, dtype=torch.float32, device=device)
        yall = torch.as_tensor(y, dtype=torch.float32, device=device)
        model = TeacherMLP(
            n_genes=n_g, out_dim=out_dim,
            hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim, dropout=cfg.dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
        bs_all = min(cfg.batch_size_max, max(8, int(math.ceil(n_b / 4))))
        pos_weight_all = _bce_pos_weight(y, mode, device)
        for ep in range(final_epochs):
            model.train()
            perm_all = torch.randperm(n_b, device=device)
            for s in range(0, n_b, bs_all):
                ix = perm_all[s : s + bs_all]
                xb = Xall[ix]
                yb = yall[ix]
                opt.zero_grad()
                out, _ = model(xb)
                loss = _loss(out, yb, mode, pos_weight_all)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
        refit_done = True
    else:
        # keep the stage-1 best-on-val weights (deploy early-stopped model)
        model.load_state_dict(best_sd)

    for p in model.parameters():
        p.requires_grad = False

    log.info(
        "[teacher] mode=%s n_b=%d n_g=%d stage1_epochs=%d best_epoch=%d best %s=%.4f",
        mode, n_b, n_g, len(history), best_epoch, metric_name, best,
    )
    return TeacherFitResult(
        model=model,
        metric_name=metric_name,
        val_metric=float(best),
        out_dim=out_dim,
        cls_pos_index=pos_idx,
        val_history=history,
        refit_full=refit_done,
        best_epoch=int(best_epoch),
        final_epochs=int(final_epochs) if refit_done else len(history),
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
