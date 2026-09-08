from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

VISIUM_SPOT_UM = 55.0

MICRON_BY_CONVENTION = {"xenium", "atera", "openst", "generic", "visium_hd"}

def median_nearest_neighbour(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        raise ValueError("need at least two units to measure a pitch")
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))

def um_per_unit(xy: np.ndarray, platform: str, spatial_uns=None, log=None,
                section=None) -> float:
    platform = platform.lower()
    if platform in MICRON_BY_CONVENTION:
        if log is not None:
            log.record("coord_scale_source", "convention", section="geometry")
            log.record("um_per_unit", 1.0, section="geometry")
        return 1.0
    if platform != "visium":
        raise ValueError(
            f"platform {platform!r} has no coordinate convention here; add one "
            f"rather than assuming the stored units are micrometres"
        )

    stored = _stored_spot_diameter(spatial_uns)
    if stored is None or not np.isfinite(stored) or stored <= 0:
        raise ValueError(
            "a Visium slide needs spot_diameter_fullres in "
            "uns['spatial'][...]['scalefactors']; without it the stored "
            "coordinates cannot be put on a physical scale, and this slide "
            "has none"
        )
    scale = VISIUM_SPOT_UM / float(stored)

    if log is not None:
        pitch = _sectioned_pitch(xy, section)
        log.record("coord_scale_source", "stored_spot_diameter", section="geometry")
        log.record("um_per_unit", scale, section="geometry")
        log.record("stored_spot_diameter_fullres", float(stored), section="geometry")
        log.record("median_nearest_neighbour_stored", pitch, section="geometry")

        log.record("stored_diameter_over_measured_pitch",
                   float(stored) / pitch if pitch > 0 else float("nan"),
                   section="geometry")
        span = np.ptp(np.asarray(xy, float), axis=0) * scale / 1000.0
        log.record("slide_extent_mm", [round(float(v), 2) for v in span],
                   section="geometry")
    return scale

def _sectioned_pitch(xy, section) -> float:
    xy = np.asarray(xy, dtype=float)
    if section is None:
        return median_nearest_neighbour(xy)
    section = np.asarray(section)
    per = []
    for value in np.unique(section):
        take = section == value
        if take.sum() >= 2:
            per.append(median_nearest_neighbour(xy[take]))
    return float(np.median(per)) if per else float("nan")

def _stored_spot_diameter(spatial_uns) -> float | None:
    if not isinstance(spatial_uns, dict):
        return None
    for entry in spatial_uns.values():
        sf = entry.get("scalefactors") if isinstance(entry, dict) else None
        if isinstance(sf, dict) and "spot_diameter_fullres" in sf:
            return float(sf["spot_diameter_fullres"])
    return None

def neighbour_graph(xy: np.ndarray, radius_um: float, um_per_unit_: float,
                    section=None) -> sparse.csr_matrix:
    xy = np.asarray(xy, dtype=float)
    radius = float(radius_um) / float(um_per_unit_)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    n = xy.shape[0]
    if section is not None:
        section = np.asarray(section)
        if pairs.size:
            pairs = pairs[section[pairs[:, 0]] == section[pairs[:, 1]]]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)]) if pairs.size \
        else np.arange(n)
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)]) if pairs.size \
        else np.arange(n)
    return sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))

def neighbour_counts(adj: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(adj.sum(axis=1)).ravel().astype(int) - 1

def check_neighbourhood(adj, radius_um: float, expected: tuple, log=None) -> None:
    deg = neighbour_counts(adj)
    med = int(np.median(deg))
    if log is not None:
        log.record("neighbour_median", med, section="geometry")
        log.record("neighbour_min", int(deg.min()), section="geometry")
        log.record("neighbour_max", int(deg.max()), section="geometry")
        log.record("fraction_isolated", float((deg == 0).mean()), section="geometry")
    lo, hi = expected
    if log is not None:
        surprising = not (lo <= med <= hi)
        log.record("neighbour_count_surprising", bool(surprising), section="geometry")
        if surprising:
            log.record("neighbour_count_note",
                       f"median {med} at {radius_um:g} um, expected {lo}-{hi}; "
                       f"the radius is a fixed physical distance and is not adjusted",
                       section="geometry")

def niche_profile(values: np.ndarray, adj: sparse.csr_matrix,
                  weights: np.ndarray):
    values = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != values.shape[0]:
        raise ValueError(f"weights has {w.shape[0]} rows but values has {values.shape[0]}")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    off = sparse.csr_matrix(adj).copy()
    off.setdiag(0)
    off.eliminate_zeros()
    den = np.asarray(off @ w).ravel()
    has = den > 0
    out = np.full(values.shape, np.nan)
    num = off @ (values * w[:, None])
    out[has] = np.asarray(num)[has] / den[has, None]
    return out, has

def niche_mean(values: np.ndarray, adj: sparse.csr_matrix,
               weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != values.shape[0]:
        raise ValueError(f"weights has {w.shape[0]} rows but values has {values.shape[0]}")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    num = adj @ (values * w[:, None])
    den = np.asarray(adj @ w).ravel()
    return np.asarray(num) / den[:, None]
