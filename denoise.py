from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

GRAPHVAE_REPRODUCIBLE = False
SMOOTH_REPRODUCIBLE = True

DENSE_BUDGET_GB = 8.0

MODES = ("none", "smooth", "graphvae")

@dataclass(frozen=True)
class GraphVAEConfig:

    hidden_dim: int = 256
    latent_dim: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    kl_weight: float = 1e-7
    val_frac: float = 0.15
    weight_decay: float = 1e-5
    gene_chunk: int = 4096

def check_dense_budget(n_units: int, n_genes: int, limit_gb: float = DENSE_BUDGET_GB):
    gb = n_units * n_genes * 4 / 1024 ** 3
    if gb > limit_gb:
        raise MemoryError(
            f"denoising would materialise a dense {n_units:,} x {n_genes:,} "
            f"float32 matrix = {gb:.1f} GB, over the {limit_gb:.0f} GB budget. "
            f"This path is for spot resolution; a cell-resolution slide has to "
            f"be read on the gene-set axes instead."
        )
    return gb

def _edges(adj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = sparse.csr_matrix(adj).copy()
    off.setdiag(0)
    off.eliminate_zeros()
    coo = off.tocoo()
    if coo.nnz == 0:
        raise ValueError(
            "the spatial graph has no neighbour edges, so there is nothing to "
            "denoise against. Check the radius and the coordinate units rather "
            "than proceeding without denoising."
        )
    deg = np.asarray(off.sum(axis=1)).ravel()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), deg

def neighbour_dispersion(x, adj) -> float:
    src, dst, _ = _edges(adj)
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean((x[src] - x[dst]) ** 2))

def smooth_denoise(x, adj, library, log=None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    _edges(adj)
    closed = sparse.csr_matrix(adj).copy()
    closed.setdiag(1.0)
    w = np.asarray(library, dtype=np.float64).ravel()
    if w.size != x.shape[0]:
        raise ValueError(f"library has {w.size} entries for {x.shape[0]} units")
    num = closed @ (x * w[:, None])
    den = np.asarray(closed @ w).ravel()
    out = np.asarray(num / np.maximum(den, 1e-300)[:, None], dtype=np.float32)
    if log is not None:
        _record(log, "smooth", adj, SMOOTH_REPRODUCIBLE, x, out, extra={})
    return out

def _record(log, mode, adj, reproducible, before, after, extra):
    src, _, _ = _edges(adj)
    log.record("mode", mode, section="denoise")
    log.record("reproducible", bool(reproducible), section="denoise")
    log.record("n_edges", int(src.size), section="denoise")
    log.record("neighbour_dispersion_before",
               round(neighbour_dispersion(before, adj), 6), section="denoise")
    log.record("neighbour_dispersion_after",
               round(neighbour_dispersion(after, adj), 6), section="denoise")
    for k, v in extra.items():
        log.record(k, v, section="denoise")

def graphvae_denoise(x, adj, cfg: GraphVAEConfig | None = None, seed: int = 42,
                     log=None) -> np.ndarray:
    import torch
    import torch.nn as nn

    cfg = cfg or GraphVAEConfig()
    x = np.asarray(x, dtype=np.float32)
    n, g = x.shape
    check_dense_budget(n, g)
    src, dst, deg = _edges(adj)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    src_t = torch.as_tensor(src, device=device)
    dst_t = torch.as_tensor(dst, device=device)
    deg_t = torch.as_tensor(np.maximum(deg, 1.0), dtype=torch.float32,
                            device=device).unsqueeze(1)

    class _Sage(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin_self = nn.Linear(d_in, d_out)
            self.lin_neigh = nn.Linear(d_in, d_out)

        def forward(self, h):
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_t, h[src_t])
            return self.lin_self(h) + self.lin_neigh(agg / deg_t)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            h, z = cfg.hidden_dim, cfg.latent_dim
            self.in_proj = nn.Linear(g, h)
            self.g1, self.g2 = _Sage(h, h), _Sage(h, h)
            self.drop = nn.Dropout(cfg.dropout)
            self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
            self.dec = nn.Sequential(nn.Linear(z, h), nn.LayerNorm(h), nn.GELU(),
                                     nn.Dropout(cfg.dropout), nn.Linear(h, g))

        def encode(self, inp):
            h = self.in_proj(inp)
            h = self.drop(torch.relu(self.g1(h)))
            h = torch.relu(self.g2(h))
            return self.mu(h), self.lv(h)

    net = _Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_val = max(1, int(round(cfg.val_frac * n))) if n > 1 else 0
    val = torch.as_tensor(np.sort(order[:n_val]), device=device)
    tr = torch.as_tensor(np.sort(order[n_val:]), device=device)

    def _recon_mse(z, rows):
        hidden = net.dec[:-1](z[rows])
        total, seen = 0.0, 0
        for start in range(0, g, cfg.gene_chunk):
            stop = min(start + cfg.gene_chunk, g)
            w = net.dec[-1].weight[start:stop]
            b = net.dec[-1].bias[start:stop]
            total = total + ((hidden @ w.T + b - X[rows, start:stop]) ** 2).sum()
            seen += (stop - start) * rows.numel()
        return total / max(seen, 1)

    best, best_state, wait, epochs = np.inf, None, 0, 0
    for epochs in range(1, cfg.max_epochs + 1):
        net.train()
        opt.zero_grad()
        mu, lv = net.encode(X)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        kl = -0.5 * (1 + lv[tr] - mu[tr].pow(2) - lv[tr].exp()).mean()
        (_recon_mse(z, tr) + cfg.kl_weight * kl).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            mu_e, _ = net.encode(X)
            v = float(_recon_mse(mu_e, val)) if n_val else float(_recon_mse(mu_e, tr))
        if v < best - 1e-9:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    out = np.empty((n, g), dtype=np.float32)
    with torch.no_grad():
        mu_e, _ = net.encode(X)
        for start in range(0, g, cfg.gene_chunk):
            stop = min(start + cfg.gene_chunk, g)
            w = net.dec[-1].weight[start:stop]
            b = net.dec[-1].bias[start:stop]
            chunk = net.dec[:-1](mu_e) @ w.T + b

            out[:, start:stop] = torch.clamp(chunk, min=0.0).cpu().numpy()

    if log is not None:
        _record(log, "graphvae", adj, GRAPHVAE_REPRODUCIBLE, x, out,
                extra={"device": device.type,
                       "hidden_dim": cfg.hidden_dim,
                       "latent_dim": cfg.latent_dim,
                       "kl_weight": cfg.kl_weight,
                       "epochs_run": int(epochs),
                       "val_mse": round(float(best), 6),
                       "n_held_out": int(n_val)})
    return out

def denoise(mode: str, x, adj, library, cfg: GraphVAEConfig | None = None,
            seed: int = 42, log=None) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown denoise mode {mode!r}; expected one of {MODES}")
    if mode == "none":
        if log is not None:
            log.record("mode", "none", section="denoise")
            log.record("reproducible", True, section="denoise")
        return np.asarray(x, dtype=np.float32)
    if mode == "smooth":
        return smooth_denoise(x, adj, library, log=log)
    return graphvae_denoise(x, adj, cfg, seed=seed, log=log)
