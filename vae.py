"""
Optional conditional VAE for target-side technical batch correction.

Invoked when ``mapper(..., batch_key=...)`` or
``mapper_multi_slide(..., batch_key=...)``.

Caller note: passing a ``batch_key`` whose values are 1:1 with ``slide_id``
will let the conditional VAE erase ALL cross-slide variation -- including
real biology (e.g. tumor content differences across slides).  The caller
is responsible for ensuring slides are technical replicates in that case.
``fit_conditional_vae`` will emit a warning when it detects this situation
(via the optional ``slide_ids`` argument) but will NOT block execution.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ._internal_config import VAEConfig, NBVAEConfig, GraphVAEConfig


class _ConditionalVAE(nn.Module):
    def __init__(self, n_genes: int, n_batches: int, h: int, z: int, dropout: float):
        super().__init__()
        self.n_batches = n_batches
        self.enc = nn.Sequential(
            nn.Linear(n_genes + n_batches, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
        )
        self.mu = nn.Linear(h, z)
        self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(
            nn.Linear(z + n_batches, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_genes),
        )

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x, b_oh):
        xb = torch.cat([x, b_oh], dim=-1)
        h = self.enc(xb)
        mu = self.mu(h); lv = self.lv(h)
        z = self.reparam(mu, lv)
        zb = torch.cat([z, b_oh], dim=-1)
        x_rec = self.dec(zb)
        return x_rec, mu, lv, z


def _batch_aligns_with_slide(batch_labels: np.ndarray,
                             slide_ids: np.ndarray) -> bool:
    """True iff batch_labels and slide_ids define the same partition
    of cells (i.e. each slide maps to exactly one batch and vice-versa).
    """
    if batch_labels.shape[0] != slide_ids.shape[0]:
        return False
    # codes -> compare partition equality
    _, bi = np.unique(batch_labels, return_inverse=True)
    _, si = np.unique(slide_ids, return_inverse=True)
    pair = bi.astype(np.int64) * (si.max() + 1) + si.astype(np.int64)
    n_pairs = np.unique(pair).size
    return n_pairs == np.unique(bi).size == np.unique(si).size


def fit_conditional_vae(
    target_X: np.ndarray,
    batch_labels: np.ndarray,
    *,
    slide_ids: Optional[np.ndarray] = None,
    config: Optional[VAEConfig] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    cfg = config or VAEConfig()
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if slide_ids is not None and _batch_aligns_with_slide(batch_labels, slide_ids):
        log.warning("[vae] batch_labels are 1:1 with slide_ids; the conditional "
                    "VAE will erase ALL cross-slide variation, including real "
                    "biology (e.g. tumor content). Proceed only if slides are "
                    "technical replicates.")

    N, G = target_X.shape
    uniq, inv = np.unique(batch_labels, return_inverse=True)
    B = len(uniq)
    if B < 2:
        log.info("[vae] only one batch -- skipping correction.")
        return target_X
    one_hot = np.zeros((N, B), dtype=np.float32)
    one_hot[np.arange(N), inv] = 1.0
    ref_idx = int(np.argmax(np.bincount(inv)))
    log.info("[vae] N=%d G=%d B=%d ref_batch=%s", N, G, B, uniq[ref_idx])

    X = torch.as_tensor(target_X, dtype=torch.float32, device=device)
    OH = torch.as_tensor(one_hot, dtype=torch.float32, device=device)
    model = _ConditionalVAE(G, B, cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    bs = max(256, min(8192, N // 50 + 1))
    best = float("inf"); bad = 0
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        for s in range(0, N, bs):
            ix = perm[s : s + bs]
            xb = X[ix]; ob = OH[ix]
            opt.zero_grad()
            x_rec, mu, lv, _ = model(xb, ob)
            recon = ((xb - x_rec) ** 2).mean()
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = recon + cfg.kl_weight * kl
            loss.backward(); opt.step()
            ep_loss += float(loss.detach().cpu())
        if ep_loss < best - 1e-6:
            best = ep_loss; bad = 0
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_sd)

    # decode all cells with the reference batch one-hot
    model.eval()
    ref_oh = torch.zeros((N, B), dtype=torch.float32, device=device)
    ref_oh[:, ref_idx] = 1.0
    out = np.empty((N, G), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            xb = X[s:e]; ob = OH[s:e]
            h = model.enc(torch.cat([xb, ob], dim=-1))
            mu = model.mu(h)
            zr = mu  # use mean (deterministic)
            zb = torch.cat([zr, ref_oh[s:e]], dim=-1)
            x_corr = model.dec(zb)
            out[s:e] = x_corr.cpu().numpy()
    log.info("[vae] correction done.")
    return out


# ---------------------------------------------------------------------------
# NB-decoder VAE for scRNA-mode reconstruction (scVI-lite)
# ---------------------------------------------------------------------------
class _NBVAE(nn.Module):
    """VAE with a negative-binomial decoder, parameterised by (mean, dispersion).

    Encoder input is log1p(CPM)-normalized expression.  Decoder produces a
    gene-proportion vector (softmax) which is scaled by per-cell library size
    to give the NB mean.  Dispersion is per-gene (learnable).
    """

    def __init__(self, n_genes: int, h: int, z: int, dropout: float):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_genes, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
        )
        self.mu = nn.Linear(h, z)
        self.lv = nn.Linear(h, z)
        self.dec_hidden = nn.Sequential(
            nn.Linear(z, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dec_rho = nn.Linear(h, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x_log, lib_size):
        h = self.enc(x_log)
        mu = self.mu(h); lv = self.lv(h)
        z = self.reparam(mu, lv)
        d = self.dec_hidden(z)
        rho = torch.softmax(self.dec_rho(d), dim=-1)
        mean = rho * lib_size.unsqueeze(-1)
        theta = self.log_theta.exp()
        return mean, theta, mu, lv


def _nb_nll(x: torch.Tensor, mean: torch.Tensor, theta: torch.Tensor,
            eps: float = 1e-8) -> torch.Tensor:
    """Negative-binomial negative log-likelihood (mean-dispersion form)."""
    log_theta_eps = torch.log(theta + eps)
    log_mu_theta = torch.log(mean + theta + eps)
    nll = (
        torch.lgamma(theta + eps)
        + torch.lgamma(x + 1.0)
        - torch.lgamma(x + theta + eps)
        + theta * (log_mu_theta - log_theta_eps)
        + x * (log_mu_theta - torch.log(mean + eps))
    )
    return nll.mean()


def fit_nb_vae(
    counts_X: np.ndarray,
    *,
    config: Optional[NBVAEConfig] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Train NB-decoder VAE on raw counts; return log1p(CPM)-denoised expression."""
    cfg = config or NBVAEConfig()
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, G = counts_X.shape
    counts_X = np.asarray(counts_X, dtype=np.float32)
    lib = counts_X.sum(axis=1).astype(np.float32)
    lib_safe = np.maximum(lib, 1.0)
    log.info("[nbvae] N=%d G=%d hidden=%d latent=%d", N, G, cfg.hidden_dim, cfg.latent_dim)

    model = _NBVAE(G, cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    # Keep counts on CPU; transfer per-batch to save GPU memory (full matrix is
    # ~600 MB for 5k cells × 35k genes; we duplicate it once for the log1p input).
    Xc_cpu = torch.as_tensor(counts_X, dtype=torch.float32)
    Lib_cpu = torch.as_tensor(lib_safe, dtype=torch.float32)
    if device.type == "cuda":
        Xc_cpu = Xc_cpu.pin_memory()
        Lib_cpu = Lib_cpu.pin_memory()

    bs = cfg.batch_size
    best = float("inf"); bad = 0
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(N)
        ep_loss = 0.0; nb = 0
        for s in range(0, N, bs):
            ix = perm[s:s + bs]
            xb = Xc_cpu[ix].to(device, non_blocking=True)
            lib_b = Lib_cpu[ix].to(device, non_blocking=True)
            xb_log = torch.log1p(xb * (1e4 / lib_b.unsqueeze(-1)))
            opt.zero_grad()
            mean, theta, mu, lv = model(xb_log, lib_b)
            theta_full = theta.unsqueeze(0).expand_as(mean)
            recon = _nb_nll(xb, mean, theta_full)
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = recon + cfg.kl_weight * kl
            loss.backward(); opt.step()
            ep_loss += float(loss.detach().cpu()); nb += 1
        ep_loss /= max(nb, 1)
        if ep_loss < best - 1e-6:
            best = ep_loss; bad = 0
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_sd)

    model.eval()
    out = np.empty((N, G), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            xb = Xc_cpu[s:e].to(device, non_blocking=True)
            lib_b = Lib_cpu[s:e].to(device, non_blocking=True)
            xb_log = torch.log1p(xb * (1e4 / lib_b.unsqueeze(-1)))
            h = model.enc(xb_log)
            mu = model.mu(h)
            d = model.dec_hidden(mu)
            rho = torch.softmax(model.dec_rho(d), dim=-1)
            mean_cpm = rho * 1e4
            out[s:e] = torch.log1p(mean_cpm).cpu().numpy()
    log.info("[nbvae] done. best_loss=%.4f", best)
    return out


# ---------------------------------------------------------------------------
# Spatial GraphVAE for stRNA-mode reconstruction (STAGATE-lite)
# ---------------------------------------------------------------------------
class _GraphSAGELayer(nn.Module):
    """Mean-aggregation SAGE layer (mirrors student.SimpleSAGEConv)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return self.lin_self(x)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        cnt = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, x[src])
        cnt.index_add_(0, dst, torch.ones((src.size(0), 1), dtype=x.dtype, device=x.device))
        agg = agg / cnt.clamp(min=1)
        return self.lin_self(x) + self.lin_neigh(agg)


class _GraphVAE(nn.Module):
    """Spatial GraphVAE: gene→hidden projection, GraphSAGE encoder, linear decoder.

    Note: the gene→hidden projection happens *before* any graph aggregation.
    This avoids materialising an ``(n_edges, n_genes)`` tensor in the SAGE
    layer (which would OOM at ~9 GiB for 70k edges × 35k genes).  All graph
    propagation happens in the much smaller hidden space.
    """

    def __init__(self, n_genes: int, h: int, z: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(n_genes, h)
        self.gconv1 = _GraphSAGELayer(h, h)
        self.gconv2 = _GraphSAGELayer(h, h)
        self.dropout = nn.Dropout(dropout)
        self.mu = nn.Linear(h, z)
        self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(
            nn.Linear(z, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_genes),
        )

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x, edge_index):
        h = torch.relu(self.in_proj(x))
        h = self.gconv1(h, edge_index); h = torch.relu(h); h = self.dropout(h)
        h = self.gconv2(h, edge_index); h = torch.relu(h)
        mu = self.mu(h); lv = self.lv(h)
        z = self.reparam(mu, lv)
        x_rec = self.dec(z)
        return x_rec, mu, lv


def _graphvae_encode(model: "_GraphVAE", x: torch.Tensor, ei: torch.Tensor):
    """Run only the encoder (returns mu, lv).  Decoder is applied separately."""
    h = torch.relu(model.in_proj(x))
    h = model.gconv1(h, ei); h = torch.relu(h); h = model.dropout(h)
    h = model.gconv2(h, ei); h = torch.relu(h)
    mu = model.mu(h); lv = model.lv(h)
    return mu, lv


def _decode_mse_chunked(
    model: "_GraphVAE",
    z: torch.Tensor,
    x_target: torch.Tensor,
    gene_chunk: int = 4096,
) -> torch.Tensor:
    """Compute MSE between dec(z) and x_target without ever materialising the
    full (N, G) reconstruction.  Decodes one gene-block at a time."""
    # Run the hidden part of the decoder once
    *hidden_layers, last_linear = list(model.dec)
    h_dec = z
    for layer in hidden_layers:
        h_dec = layer(h_dec)
    W = last_linear.weight   # (G, h)
    b = last_linear.bias     # (G,)
    n_genes = W.size(0)
    total_se = h_dec.new_zeros(())
    count = 0
    for g_start in range(0, n_genes, gene_chunk):
        g_end = min(g_start + gene_chunk, n_genes)
        x_rec_chunk = h_dec @ W[g_start:g_end].t() + b[g_start:g_end]
        diff = x_target[:, g_start:g_end] - x_rec_chunk
        total_se = total_se + (diff * diff).sum()
        count += diff.numel()
    return total_se / max(count, 1)


def _fit_graph_vae_impl(
    target_X: np.ndarray,
    edge_index: torch.Tensor,
    cfg: GraphVAEConfig,
    device: torch.device,
    log: logging.Logger,
) -> np.ndarray:
    N, G = target_X.shape
    log.info("[graphvae] N=%d G=%d hidden=%d latent=%d edges=%d device=%s",
             N, G, cfg.hidden_dim, cfg.latent_dim, int(edge_index.shape[1]), device)
    model = _GraphVAE(G, cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    X = torch.as_tensor(target_X, dtype=torch.float32, device=device)
    ei = edge_index.long().to(device)
    best = float("inf"); bad = 0
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        mu, lv = _graphvae_encode(model, X, ei)
        z = model.reparam(mu, lv)
        recon = _decode_mse_chunked(model, z, X)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        loss = recon + cfg.kl_weight * kl
        loss.backward(); opt.step()
        cur = float(loss.detach().cpu())
        if cur < best - 1e-6:
            best = cur; bad = 0
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        mu, _ = _graphvae_encode(model, X, ei)
        # Decode chunked into preallocated output (no full (N,G) tensor at once).
        out = np.empty((N, G), dtype=np.float32)
        h_dec = mu
        for layer in list(model.dec)[:-1]:
            h_dec = layer(h_dec)
        last = list(model.dec)[-1]
        gene_chunk = 4096
        for g_start in range(0, G, gene_chunk):
            g_end = min(g_start + gene_chunk, G)
            x_rec_chunk = h_dec @ last.weight[g_start:g_end].t() + last.bias[g_start:g_end]
            x_rec_chunk = torch.clamp(x_rec_chunk, min=0.0)
            out[:, g_start:g_end] = x_rec_chunk.cpu().numpy()
    log.info("[graphvae] done. best_loss=%.4f", best)
    return out


def fit_graph_vae(
    target_X: np.ndarray,
    edge_index: torch.Tensor,
    *,
    config: Optional[GraphVAEConfig] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Train spatial GraphVAE on log1p(CPM) and return reconstructed expression.

    Falls back to CPU automatically if a CUDA OOM occurs.
    """
    cfg = config or GraphVAEConfig()
    log = logger or logging.getLogger("MAPPER")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return _fit_graph_vae_impl(target_X, edge_index, cfg, device, log)
    except torch.cuda.OutOfMemoryError:
        log.warning("[graphvae] CUDA OOM; retrying on CPU.")
        torch.cuda.empty_cache()
        return _fit_graph_vae_impl(target_X, edge_index, cfg,
                                   torch.device("cpu"), log)
