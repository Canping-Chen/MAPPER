"""
Student GNN — *optional* niche-score distillation.

Default MAPPER pipeline uses the pseudo-bulk-through-MLP route to compute
``s_niche`` directly (see ``mapper.niche.compute_s_niche_direct``).  That route
is non-linear, interpretable, and on the same scale as bulk training scores.

The Student GNN is offered as an *advanced* alternative for two scenarios:

  1. Inference on a new slide where re-computing the LOO pseudo-bulk is
     expensive (e.g. streaming, large multi-slide deployment).  Once trained
     the GNN takes only ``(teacher_emb, G_spatial)`` and produces ``s_gnn``
     in a single forward pass.

  2. Methodological comparisons / ablations.

Training objective
------------------
The GNN distills the pseudo-bulk niche score:

    s_gnn  = StudentGNN(teacher_emb, G_spatial)
    loss   = MSE(s_gnn, s_niche_pseudobulk) + lambda_smooth * Lap(s_gnn, G_spatial)

The first SAGE layer uses ``include_self=False`` so the first-hop
representation is built only from spatial neighbours; the second layer keeps
the self path to combine refined neighbourhood and intrinsic information.
"""
from __future__ import annotations
import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._internal_config import StudentConfig
from .graph import laplacian_smoothness_loss


# ---------------------------------------------------------------------------
# minimal SAGE conv (no torch_geometric dependency)
# ---------------------------------------------------------------------------
class SimpleSAGEConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        include_self: bool = True,
        edge_chunk: int = 2_000_000,
    ):
        super().__init__()
        self.include_self = bool(include_self)
        self.lin_self = nn.Linear(in_dim, out_dim) if include_self else None
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.edge_chunk = int(edge_chunk)

    def _aggregate(self, x, edge_index, edge_weight):
        src = edge_index[0]; dst = edge_index[1]
        E = src.size(0)
        agg = torch.zeros_like(x)
        if edge_weight is None:
            cnt = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
            for s in range(0, E, self.edge_chunk):
                e = min(s + self.edge_chunk, E)
                src_c = src[s:e]; dst_c = dst[s:e]
                agg.index_add_(0, dst_c, x.index_select(0, src_c))
                cnt.index_add_(
                    0, dst_c,
                    torch.ones((e - s, 1), dtype=x.dtype, device=x.device),
                )
            return agg / cnt.clamp(min=1)
        ew = edge_weight.to(x.dtype).view(-1, 1)
        wsum = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        for s in range(0, E, self.edge_chunk):
            e = min(s + self.edge_chunk, E)
            src_c = src[s:e]; dst_c = dst[s:e]; ew_c = ew[s:e]
            agg.index_add_(0, dst_c, x.index_select(0, src_c) * ew_c)
            wsum.index_add_(0, dst_c, ew_c)
        return agg / wsum.clamp(min=1e-6)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_index is None or edge_index.numel() == 0:
            if self.lin_self is not None:
                return self.lin_self(x)
            return torch.zeros(x.size(0), self.lin_neigh.out_features,
                               dtype=x.dtype, device=x.device)
        ew_in = edge_weight.detach() if edge_weight is not None else None
        ei_in = edge_index.detach()
        if self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            agg = checkpoint(self._aggregate, x, ei_in, ew_in, use_reentrant=False)
        else:
            agg = self._aggregate(x, ei_in, ew_in)
        if self.lin_self is not None:
            return self.lin_self(x) + self.lin_neigh(agg)
        return self.lin_neigh(agg)


class StudentGNN(nn.Module):
    """2-layer SAGE with scalar head.  Output is ``s_gnn`` of shape (N,)."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = SimpleSAGEConv(in_dim, hidden_dim, include_self=False)
        self.conv2 = SimpleSAGEConv(hidden_dim, out_dim, include_self=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.elu(h)
        return self.head(h).view(-1)


# ---------------------------------------------------------------------------
# distillation training (target = pseudo-bulk niche score)
# ---------------------------------------------------------------------------
def fit_student_distill(
    teacher_emb: np.ndarray,              # (N, D) — node features
    s_niche_target: np.ndarray,           # (N,)   — pseudo-bulk MLP score (the label)
    edge_index: Optional[torch.Tensor],   # (2, E) — G_spatial only
    *,
    config: Optional[StudentConfig] = None,
    device: Optional[torch.device] = None,
    random_seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Train the Student GNN to distill the pseudo-bulk niche score.

    Returns ``s_gnn`` (N,) — the GNN's prediction of the niche score, on the
    same scale as ``s_niche_target``.  Callers compose the final score as
    ``s_final = (1-alpha) * s_cell + alpha * s_gnn``.
    """
    cfg = config or StudentConfig()
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, D = teacher_emb.shape
    if edge_index is None or edge_index.numel() == 0:
        log.warning("[student] no spatial edges; returning niche target unchanged")
        return np.asarray(s_niche_target, dtype=np.float32).copy()

    log.info(
        "[student] distillation: N=%d D=%d hidden=%d out=%d λ_smooth=%.3f device=%s",
        N, D, cfg.hidden_dim, cfg.out_dim, cfg.lambda_smooth, device,
    )
    torch.manual_seed(random_seed)

    use_minibatch = N > 20_000
    if use_minibatch:
        try:
            import torch_geometric  # noqa: F401
        except Exception:
            log.warning("[student] torch_geometric unavailable -> full-graph training")
            use_minibatch = False

    model = StudentGNN(D, cfg.hidden_dim, cfg.out_dim, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    X = torch.as_tensor(teacher_emb, dtype=torch.float32)
    y = torch.as_tensor(s_niche_target, dtype=torch.float32)
    ei = edge_index.long()

    best_loss = math.inf; bad = 0
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if not use_minibatch:
        Xd = X.to(device); yd = y.to(device); eid = ei.to(device)
        for ep in range(cfg.max_epochs):
            model.train()
            opt.zero_grad()
            pred = model(Xd, eid)
            mse = F.mse_loss(pred, yd)
            sm = laplacian_smoothness_loss(pred, eid)
            loss = mse + cfg.lambda_smooth * sm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            cur = float(loss.detach().cpu())
            if cur < best_loss - 1e-6:
                best_loss = cur; bad = 0
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_sd)
        with torch.no_grad():
            out = model(Xd, eid).detach().cpu().numpy()
        log.info("[student] full-graph done. best_loss=%.4f", best_loss)
        return out.astype(np.float32)

    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    data = Data(x=X, edge_index=ei, y=y)
    train_loader = NeighborLoader(
        data, num_neighbors=list(cfg.neighbor_sizes),
        batch_size=cfg.minibatch_size, shuffle=True,
        num_workers=8, persistent_workers=True,
    )
    for ep in range(cfg.max_epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index)
            target = batch.y[: batch.batch_size]
            pred_b = pred[: batch.batch_size]
            mse = F.mse_loss(pred_b, target)
            sm = laplacian_smoothness_loss(pred, batch.edge_index)
            loss = mse + cfg.lambda_smooth * sm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach().cpu()); nb += 1
        ep_loss /= max(nb, 1)
        if ep_loss < best_loss - 1e-6:
            best_loss = ep_loss; bad = 0
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_sd)

    model.eval()
    out = np.empty(N, dtype=np.float32)
    inf_loader = NeighborLoader(
        data, num_neighbors=list(cfg.neighbor_sizes),
        batch_size=max(cfg.minibatch_size * 2, 4096),
        shuffle=False, num_workers=8, persistent_workers=False,
    )
    with torch.no_grad():
        for batch in inf_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)
            n_id = batch.input_id if hasattr(batch, "input_id") else batch.n_id[: batch.batch_size]
            out[n_id.cpu().numpy()] = pred[: batch.batch_size].cpu().numpy()
    log.info("[student] minibatch done. best_loss=%.4f", best_loss)
    return out
