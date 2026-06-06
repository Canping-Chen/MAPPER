"""
MAPPER common utilities: backend detection, device, seeding, logging.

Backend selection follows the user's requirement:
  - GPU is auto-used when available;
  - rapids-singlecell / cuML / FAISS are used opportunistically when N >= threshold;
  - on import failure or small N, fall back to scanpy / scipy / sklearn / numpy.

The user-facing API never sees backend or device parameters.
"""
from __future__ import annotations
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
_LOGGER_NAME = "MAPPER"


def get_logger(verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# device & seeding
# ---------------------------------------------------------------------------
def auto_device():
    """Auto-pick CUDA -> MPS -> CPU. Never exposed to the user."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_best_cuda_device(min_free_gib: float = 0.0, logger=None):
    """Return the visible CUDA device with the most free memory.

    Useful for routing PyTorch work (e.g. the student GNN) to a GPU that
    hasn't been polluted by cupy / RMM / cuML pools from earlier stages —
    those allocators don't always return memory to the driver even after
    :func:`free_gpu_memory`, so on multi-GPU hosts the only practical way to
    get a clean slate is to use a different device.

    Returns CPU only if (a) no CUDA device is visible, or (b) no visible
    device has ``min_free_gib`` of free memory.  By default falls back to
    the freest GPU regardless of size.
    """
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device("cpu")
    n = torch.cuda.device_count()
    free_per_dev = []
    for i in range(n):
        try:
            with torch.cuda.device(i):
                free, _total = torch.cuda.mem_get_info()
        except Exception:
            free = 0
        free_per_dev.append(free)
    best_idx = max(range(n), key=lambda i: free_per_dev[i])
    best_free = free_per_dev[best_idx]
    if logger is not None:
        logger.info(
            "[mem] free per device: %s -> picked cuda:%d (%.2f GiB free)",
            ", ".join(f"cuda:{i}={f/2**30:.2f}GiB" for i, f in enumerate(free_per_dev)),
            best_idx, best_free / 2**30,
        )
    if min_free_gib > 0 and best_free < min_free_gib * (2 ** 30):
        if logger is not None:
            logger.warning(
                "[mem] no GPU has >= %.1f GiB free; falling back to CPU",
                min_free_gib,
            )
        return torch.device("cpu")
    return torch.device(f"cuda:{best_idx}")


def list_cuda_devices():
    """Return list of all available CUDA devices, or [auto_device()] otherwise.

    Used by the teacher-inference path to fan chunks across every visible
    GPU instead of pinning the whole target matrix to a single device.
    """
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 0:
            return [torch.device(f"cuda:{i}") for i in range(n)]
    return [auto_device()]


def free_gpu_memory(logger=None) -> None:
    """Release cached memory from torch, cupy, RMM, and rapids singlecell.

    Earlier MAPPER stages (sparse log1p on GPU via rapids_singlecell, KNN graph
    build via cuML / FAISS) hold memory in cupy / RMM pools that PyTorch can't
    see.  On a 44 GB GPU this can pin ~20 GB of "non-PyTorch" allocation across
    stages — see the OOM message: ``Of the allocated memory 1.01 GiB is
    allocated by PyTorch`` while the process holds 24 GiB.  Call this between
    major stages to hand memory back.

    NOTE: cuML keeps internal C++ state that holds references through the RMM
    allocator.  Until those references are dropped (i.e. the cuML / cupy
    objects are garbage-collected), the pool can't be fully released.  For a
    truly clean GPU after rapids stages, prefer routing later torch work to a
    different device via :func:`pick_best_cuda_device`.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    # 1. Drop cupy's cached free blocks back to RMM
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    # 2. Replace RMM's pool with the plain CUDA resource — this calls the
    #    pool's dtor, which (if no outstanding refs remain) returns memory
    #    to the driver.  Fall back to rmm.reinitialize on older RMM versions.
    try:
        import rmm
        try:
            from rmm.mr import CudaMemoryResource, set_current_device_resource
            set_current_device_resource(CudaMemoryResource())
        except Exception:
            rmm.reinitialize(pool_allocator=False, managed_memory=False)
    except Exception:
        pass
    # 3. Re-wire cupy back through the fresh RMM resource so subsequent
    #    cupy / cuML calls still work.
    try:
        import cupy as cp
        from rmm.allocators.cupy import rmm_cupy_allocator
        cp.cuda.set_allocator(rmm_cupy_allocator)
    except Exception:
        pass
    if logger is not None:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.info(
                    "[mem] cuda:%d free=%.2f GiB / %.2f GiB after cleanup",
                    torch.cuda.current_device(),
                    free / 2**30, total / 2**30,
                )
        except Exception:
            pass


def set_seed(seed: int) -> None:
    """Set seed across all relevant RNGs."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# backend detection (cached)
# ---------------------------------------------------------------------------
@dataclass
class Backends:
    cupy: bool
    rapids_singlecell: bool
    cuml: bool
    faiss: bool
    torch_cluster: bool
    torch_geometric: bool


_BACKENDS: Optional[Backends] = None


def detect_backends(force_refresh: bool = False) -> Backends:
    """Probe optional GPU / accelerated backends. Cached.

    Returns a :class:`Backends` dataclass; never raises.
    """
    global _BACKENDS
    if _BACKENDS is not None and not force_refresh:
        return _BACKENDS

    def _try(mod_name: str) -> bool:
        try:
            __import__(mod_name)
            return True
        except Exception:
            return False

    _BACKENDS = Backends(
        cupy=_try("cupy"),
        rapids_singlecell=_try("rapids_singlecell"),
        cuml=_try("cuml"),
        faiss=_try("faiss"),
        torch_cluster=_try("torch_cluster"),
        torch_geometric=_try("torch_geometric"),
    )
    return _BACKENDS


def gpu_available() -> bool:
    """Whether a CUDA GPU is available via torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def to_numpy(x):
    """Best-effort conversion to numpy array (handles torch / cupy / sparse)."""
    import scipy.sparse as sp
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if sp.issparse(x):
        return np.asarray(x.todense())
    if hasattr(x, "get"):       # cupy ndarray
        try:
            return x.get()
        except Exception:
            pass
    return np.asarray(x)


