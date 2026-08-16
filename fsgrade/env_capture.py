"""
Capture the exact software/hardware environment of a run.

Written to ``env.json`` in every run directory. A thesis reviewer asking "what
was this actually run on?" should never need to be answered from memory.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Any


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def git_info() -> dict[str, Any]:
    """Commit, dirtiness, and a hash of the uncommitted diff.

    Recording the diff hash means a run made from a dirty tree is still
    identifiable -- 'commit abc123 plus these exact uncommitted changes'.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "short": None, "dirty": None, "diff_sha256": None}

    diff = _run_git("diff", "HEAD")
    dirty = bool(diff)
    diff_hash = (
        hashlib.sha256(diff.encode("utf-8")).hexdigest() if dirty and diff else None
    )
    return {
        "commit": commit,
        "short": commit[:7],
        "dirty": dirty,
        "diff_sha256": diff_hash,
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "unknown")


def package_versions() -> dict[str, str | None]:
    names = [
        "torch", "torchvision", "numpy", "scipy", "sklearn", "pandas",
        "matplotlib", "seaborn", "PIL", "yaml", "timm", "open_clip", "tqdm",
    ]
    return {name: _version(name) for name in names}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = None
    except ImportError:  # pragma: no cover
        info["cuda_available"] = None
    return info


def relevant_env_vars() -> dict[str, str | None]:
    keys = [
        "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
        "FRUITVISION_DATA_ROOT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    ]
    return {k: os.environ.get(k) for k in keys}


def pip_freeze() -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return []


def capture_environment(*, include_pip_freeze: bool = True) -> dict[str, Any]:
    """Full environment snapshot for ``env.json``."""
    env: dict[str, Any] = {
        "schema_version": "1.0.0",
        "git": git_info(),
        "hardware": hardware_info(),
        "packages": package_versions(),
        "env_vars": relevant_env_vars(),
    }
    if include_pip_freeze:
        env["pip_freeze"] = pip_freeze()
    return env
