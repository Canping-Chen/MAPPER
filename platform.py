"""
Platform-aware coordinate handling for spatial transcriptomics.

Supported platforms
-------------------
- visium      : pixel coords; converted via ``adata.uns['spatial'][lib]['scalefactors']``.
                Default radius = 150 um.
- xenium      : already in microns. Default radius = 30 um.
- generic     : assume coords are already in microns; user must supply a radius.
- scRNA       : no spatial graph; cell-level mode.

We never expose ``coordinate_unit`` to the user; conversion is automatic.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np


PLATFORM_DEFAULT_RADIUS_UM = {
    "visium":     150.0,
    "xenium":      30.0,
    "generic":   None,        # require user-provided
    "scRNA":     None,        # no graph
}

VISIUM_SPOT_DIAMETER_UM = 65.0  # 10x Visium fixed


def is_spatial_platform(platform: str) -> bool:
    return platform in {"visium", "xenium", "generic"}


def resolve_default_radius_um(platform: str, user_radius: Optional[float]) -> Optional[float]:
    if user_radius is not None and user_radius > 0:
        return float(user_radius)
    if platform == "scRNA":
        return None
    if platform == "generic":
        return None
    return PLATFORM_DEFAULT_RADIUS_UM.get(platform, None)


def coords_to_micron(
    adata,
    platform: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Convert ``adata.obsm['spatial']`` to micron coordinates.

    Returns ``None`` if no spatial coords exist or platform is scRNA.
    """
    if platform == "scRNA":
        return None
    if "spatial" not in adata.obsm:
        if logger is not None:
            logger.warning("adata.obsm['spatial'] missing; running cell-level mode.")
        return None

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            f"adata.obsm['spatial'] must be (N, 2) or (N, 3); got {coords.shape}"
        )

    if platform == "visium":
        spot_diam_fullres = _resolve_visium_spot_diameter_fullres(adata)
        if spot_diam_fullres is None:
            if logger is not None:
                logger.warning(
                    "[platform=%s] could not resolve spot_diameter_fullres from "
                    "adata.uns['spatial']; assuming coords are already micron.",
                    platform,
                )
            return coords.copy()
        physical_diam = VISIUM_SPOT_DIAMETER_UM
        scale = float(physical_diam) / float(spot_diam_fullres)
        if logger is not None:
            logger.info(
                "[platform=%s] pixel->micron scale = %.6f (spot_diameter_fullres=%.2f, "
                "physical=%.2f um)",
                platform, scale, spot_diam_fullres, physical_diam,
            )
        return coords * scale

    if platform == "xenium":
        # Xenium coords are micron by convention.
        return coords.copy()

    # generic: assume micron
    return coords.copy()


def _resolve_visium_spot_diameter_fullres(adata) -> Optional[float]:
    spatial_uns = adata.uns.get("spatial", None)
    if not isinstance(spatial_uns, dict):
        return None
    for lib_id, payload in spatial_uns.items():
        if not isinstance(payload, dict):
            continue
        scalef = payload.get("scalefactors", None)
        if not isinstance(scalef, dict):
            continue
        v = scalef.get("spot_diameter_fullres", None)
        if v is not None and float(v) > 0:
            return float(v)
    return None
