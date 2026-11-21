from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

RADIUS_UM: Mapping[str, float | None] = {
    "visium": 150.0,
    "visium_hd": 30.0,
    "xenium": 30.0,
    "atera": 30.0,
    "openst": 80.0,
    "scrna": None,
}

RESOLUTION: Mapping[str, str] = {
    "visium": "spot",
    "visium_hd": "cell",
    "xenium": "cell",
    "atera": "cell",
    "openst": "cell",
    "scrna": "spot",   # per-gene coordinate (user override, 2026-10-08): scRNA
                        # uses the per-gene z path, not the Hallmark50 axes.
}

# Per-platform detected-gene floor: a unit detecting fewer than this many genes
# is marked unusable. Visium spots pool many cells and scRNA cells are deep, so
# both carry thousands of genes (500 removes only degenerate units); imaging /
# bead platforms (Xenium targeted panel, Atera, Open-ST, Visium-HD) detect far
# fewer genes per unit, so no floor is applied there (0 = keep every unit; user
# override, 2026-10-09).
MIN_GENES: Mapping[str, int] = {
    "visium": 500,
    "visium_hd": 0,
    "xenium": 0,
    "atera": 0,
    "openst": 0,
    "scrna": 500,
}

FREEZE_BASIS: Mapping[str, bool] = {"spot": True, "cell": True}

RESOLUTIONS = ("spot", "cell")

NICHE_WEIGHT: Mapping[str, float] = {"spot": 0.5, "cell": 0.5}

MIN_EVENTS_RUNG1 = 50

EVENTS_PER_PARAMETER = 10

def rung2_free_parameters(k: int) -> int:
    return k * (k - 1) // 2

def ladder_rung(n_events: int, k: int) -> int:
    if n_events < MIN_EVENTS_RUNG1:
        return 0
    if n_events >= EVENTS_PER_PARAMETER * rung2_free_parameters(k):
        return 2
    return 1

@dataclass(frozen=True)
class MapperConfig:

    seed: int = 42

    alpha_side: float = 0.30

    min_detection_fraction: float = 0.01
    bulk_mean_tpm_floor: float = 1.0
    drop_mt: bool = True
    drop_technical_rna: bool = True
    hallmark_min_genes: int = 5

    min_axis_reliability: float = 0.5

    panel_error_below: int = 20
    panel_warn_below: int = 500

    extra: Mapping[str, object] = field(default_factory=dict)

    def resolution(self, platform: str) -> str:
        key = platform.lower()
        if key not in RESOLUTION:
            raise ValueError(
                f"unknown platform {platform!r}; known platforms are "
                f"{sorted(RESOLUTION)}"
            )
        chosen = self.extra.get("resolution")
        if chosen is None:
            return RESOLUTION[key]
        chosen = str(chosen).lower()
        if chosen not in RESOLUTIONS:
            raise ValueError(
                f"unknown resolution {chosen!r}; expected one of {RESOLUTIONS}"
            )
        return chosen

    def reliability_floor(self, resolution: str) -> float | None:
        return self.min_axis_reliability if resolution == "cell" else None

    def min_genes(self, platform: str) -> int:
        key = platform.lower()
        if key not in MIN_GENES:
            raise ValueError(
                f"unknown platform {platform!r}; known platforms are "
                f"{sorted(MIN_GENES)}"
            )
        override = self.extra.get("min_genes")
        return int(override) if override is not None else MIN_GENES[key]

    def radius_um(self, platform: str) -> float | None:
        key = platform.lower()
        if key not in RADIUS_UM:
            raise ValueError(
                f"unknown platform {platform!r}; known platforms are "
                f"{sorted(RADIUS_UM)}"
            )
        return RADIUS_UM[key]
