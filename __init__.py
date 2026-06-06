"""
MAPPER — Mapping Associated Phenotype Programs to Expression-Resolved niches.

Public API:
    from mapper import mapper, mapper_multi_slide

Lazy-loaded so that lightweight submodules (platform, grouping, _internal_config)
remain importable without torch installed.
"""
from ._internal_config import (
    MapperConfig,
    TeacherConfig,
    GraphConfig,
    NicheConfig,
    StudentConfig,
    GroupingConfig,
    VAEConfig,
    DEFAULT,
)

__all__ = [
    "mapper",
    "mapper_multi_slide",
    "MapperConfig",
    "TeacherConfig",
    "GraphConfig",
    "NicheConfig",
    "StudentConfig",
    "GroupingConfig",
    "VAEConfig",
    "DEFAULT",
]
__version__ = "0.7.0"


def __getattr__(name):
    if name in {"mapper", "mapper_multi_slide"}:
        from .api import mapper as _mapper, mapper_multi_slide as _mms
        return {"mapper": _mapper, "mapper_multi_slide": _mms}[name]
    raise AttributeError(f"module 'mapper' has no attribute {name!r}")
