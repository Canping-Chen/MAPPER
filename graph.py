"""
Graph construction for MAPPER.

Only one graph is built in the current pipeline:

  G_spatial : physical micron-radius graph used by the niche pseudo-bulk
              and (optional) Student GNN refinement.  Spatial mode only.
              ``max_neighbors`` capped; no cross-slide edges.

``torch_cluster.radius_graph`` is used opportunistically when available;
otherwise we fall back to a chunked ``scipy.spatial.cKDTree``.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# spatial radius graph (with max_neighbors cap, optional batch=slide)
# ---------------------------------------------------------------------------
def build_spatial_graph(
    coords_um: np.ndarray,                 # (N, 2 or 3) micron coords
    radius_um: float,
    *,
    max_neighbors: int = 30,
    slide_ids: Optional[np.ndarray] = None,
    chunk: int = 50_000,
    use_torch_cluster: bool = True,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[torch.Tensor]:
    """Return edge_index of shape (2, E) or None when no edges exist.

    Edges respect ``slide_ids`` (no inter-slide edges) and are capped at
    ``max_neighbors`` per node (keeping nearest neighbours).
    """
    log = logger or logging.getLogger("MAPPER")
    if coords_um is None or radius_um is None or radius_um <= 0:
        return None
    coords_um = np.asarray(coords_um, dtype=np.float32)
    n = coords_um.shape[0]
    if n == 0:
        return None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_torch_cluster:
        try:
            from torch_cluster import radius_graph as tc_radius_graph
            coords_t = torch.as_tensor(coords_um, dtype=torch.float32, device=device)
            batch_t = None
            if slide_ids is not None:
                batch_t = torch.as_tensor(
                    _encode_groups_to_long(slide_ids), dtype=torch.long, device=device
                )
            ei = tc_radius_graph(coords_t, r=float(radius_um), batch=batch_t, loop=False)
            ei = _cap_neighbors_by_distance(ei, coords_t, max_neighbors)
            if ei.numel() == 0:
                return None
            log.info(
                "[graph.spatial] torch_cluster: N=%d r=%.1fum E=%d avg_deg=%.2f",
                n, radius_um, ei.size(1), 2 * ei.size(1) / max(n, 1),
            )
            return ei.cpu()
        except Exception as e:
            log.warning("[graph.spatial] torch_cluster failed (%s); using scipy KDTree", e)

    edges = _scipy_radius_graph(
        coords_um, radius_um, slide_ids=slide_ids,
        max_neighbors=max_neighbors, chunk=chunk, log=log,
    )
    if edges is None:
        return None
    return torch.as_tensor(edges, dtype=torch.long)


def _encode_groups_to_long(arr) -> np.ndarray:
    arr = np.asarray(arr)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64)


def _scipy_radius_graph(coords, radius, slide_ids, max_neighbors, chunk, log):
    from scipy.spatial import cKDTree

    n = coords.shape[0]
    if slide_ids is None:
        all_pairs = []
        tree = cKDTree(coords)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            block = coords[s:e]
            neigh_list = tree.query_ball_point(block, r=radius)
            for i_local, nb in enumerate(neigh_list):
                i = s + i_local
                nb = [j for j in nb if j != i]
                if max_neighbors and len(nb) > max_neighbors:
                    d2 = ((coords[nb] - coords[i]) ** 2).sum(axis=1)
                    keep = np.argsort(d2)[:max_neighbors]
                    nb = [nb[k] for k in keep]
                if not nb:
                    continue
                src = np.full(len(nb), i, dtype=np.int64)
                dst = np.asarray(nb, dtype=np.int64)
                all_pairs.append(np.stack([src, dst], axis=0))
        if not all_pairs:
            return None
        edges = np.concatenate(all_pairs, axis=1)
        log.info(
            "[graph.spatial] scipy: N=%d r=%.1fum E=%d avg_deg=%.2f",
            n, radius, edges.shape[1], 2 * edges.shape[1] / max(n, 1),
        )
        return edges

    slides = _encode_groups_to_long(slide_ids)
    uniq = np.unique(slides)
    all_pairs = []
    for s_id in uniq:
        idx = np.where(slides == s_id)[0]
        if len(idx) < 2:
            continue
        sub = coords[idx]
        sub_edges = _scipy_radius_graph(
            sub, radius, slide_ids=None, max_neighbors=max_neighbors, chunk=chunk, log=log
        )
        if sub_edges is None:
            continue
        all_pairs.append(idx[sub_edges])
    if not all_pairs:
        return None
    edges = np.concatenate(all_pairs, axis=1)
    log.info(
        "[graph.spatial] scipy(multi-slide): N=%d slides=%d E=%d",
        coords.shape[0], len(uniq), edges.shape[1],
    )
    return edges


def _cap_neighbors_by_distance(ei: torch.Tensor, coords: torch.Tensor, k: int) -> torch.Tensor:
    if ei.numel() == 0:
        return ei
    src = ei[0]; dst = ei[1]
    diff = coords[src] - coords[dst]
    d2 = (diff * diff).sum(dim=1)
    order = torch.argsort(d2)
    src_s = src[order]; dst_s = dst[order]
    keep = torch.ones_like(src_s, dtype=torch.bool)
    counts = torch.zeros(int(src.max().item()) + 1, dtype=torch.long, device=ei.device)
    for i in range(src_s.size(0)):
        s_i = int(src_s[i].item())
        if counts[s_i] >= k:
            keep[i] = False
        else:
            counts[s_i] += 1
    return torch.stack([src_s[keep], dst_s[keep]], dim=0)


# ---------------------------------------------------------------------------
# graph-Laplacian smoothness loss (used by Student GNN)
# ---------------------------------------------------------------------------
def laplacian_smoothness_loss(
    score: torch.Tensor,            # (N,)
    edge_index: torch.Tensor,       # (2, E)
    edge_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return score.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    diff = score[src] - score[dst]
    if edge_weight is not None:
        return (edge_weight.to(diff.dtype) * diff.pow(2)).mean() / (score.var() + eps)
    return diff.pow(2).mean() / (score.var() + eps)
