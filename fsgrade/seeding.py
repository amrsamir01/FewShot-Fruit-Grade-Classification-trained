"""
Deterministic seeding.

The original pipeline called ``set_seed(42)`` once inside notebook cell 1 and
then relied on the *global* ``random`` module for episode sampling. That makes
every result a function of cell execution order: re-running one cell shifts the
RNG stream and silently changes which episodes each method saw.

Here, seeds are *derived* rather than shared. ``derive_seed`` maps a set of
string/int components onto a stable 32-bit seed via BLAKE2b, so an episode's
randomness depends only on its identity -- never on how many episodes or
methods ran before it.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

try:
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False


_UINT32_MAX = 2**32 - 1


def derive_seed(*components: Any) -> int:
    """Map arbitrary components onto a stable 32-bit seed.

    Stable across processes, machines and Python versions (unlike ``hash()``,
    which is randomised per-process unless PYTHONHASHSEED is pinned).

    >>> derive_seed("loso-mango", "test", 42) == derive_seed("loso-mango", "test", 42)
    True
    """
    joined = "|".join(str(c) for c in components)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _UINT32_MAX


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG in play and optionally enforce deterministic kernels.

    Returns a report dict that is written into ``env.json`` so a run records
    what determinism guarantees were actually in force -- not merely requested.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "deterministic_requested": deterministic,
        "pythonhashseed": str(seed),
        "torch_available": _HAVE_TORCH,
    }

    if not _HAVE_TORCH:
        return report

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        # The original set_seed omitted this: with benchmark=True, cuDNN picks
        # algorithms by timing, which can vary run to run even when seeded.
        torch.backends.cudnn.benchmark = False
        # Required for deterministic CUBLAS GEMMs on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["use_deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover - version dependent
            report["use_deterministic_algorithms"] = f"unavailable: {exc}"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        report["use_deterministic_algorithms"] = False

    report["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return report


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from torch's per-worker base seed."""
    if not _HAVE_TORCH:  # pragma: no cover
        return
    base = torch.initial_seed() % _UINT32_MAX
    seed = derive_seed(base, worker_id)
    random.seed(seed)
    np.random.seed(seed)


def rng_for(*components: Any) -> np.random.Generator:
    """A fresh, independent NumPy generator keyed to ``components``.

    Preferred over touching the global RNG anywhere in this codebase.
    """
    return np.random.default_rng(derive_seed(*components))
