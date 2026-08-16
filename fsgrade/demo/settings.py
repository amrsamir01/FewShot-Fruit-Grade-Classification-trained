"""
Demo settings, and the offline switches.

Importing this module sets the Hugging Face / torch cache environment variables
**before** anything imports ``open_clip`` or ``timm``. Both read those variables
at import time, so setting them later has no effect — and a viva room is
precisely where a silent weight download must not happen.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
#  Offline enforcement. This runs at import, deliberately.
# --------------------------------------------------------------------------- #

_CACHE_ROOT = pathlib.Path(
    os.environ.get("FSGRADE_DEMO_CACHE", pathlib.Path.cwd() / ".cache")
).expanduser()


def enforce_offline(cache_root: pathlib.Path = _CACHE_ROOT, *, offline: bool = True) -> dict[str, str]:
    """Pin model caches locally and disable outbound calls.

    Returns the variables that were set, so ``/api/health`` can report them.
    """
    hf_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)

    applied = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TORCH_HOME": str(torch_home),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    if offline:
        applied["HF_HUB_OFFLINE"] = "1"
        applied["TRANSFORMERS_OFFLINE"] = "1"

    for key, value in applied.items():
        os.environ.setdefault(key, value)
    return applied


# Applied on import. `prefetch.py` deliberately re-runs it with offline=False.
OFFLINE_ENV = enforce_offline(offline=os.environ.get("FSGRADE_DEMO_ONLINE") != "1")


@dataclass
class DemoSettings:
    """Everything the demo needs to start."""

    results_root: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))
    data_root: pathlib.Path | None = None
    cache_root: pathlib.Path = field(default_factory=lambda: _CACHE_ROOT)
    device: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    clip_encoder: str = "clip_vitb16"
    classes: tuple[str, ...] = ("fresh", "rotten")
    species: tuple[str, ...] = ("apple", "banana", "grape", "mango", "orange")
    session_ttl: int = 3600
    max_sessions: int = 32
    model_cache_size: int = 4
    offline: bool = True

    # ------------------------------------------------------------------ #
    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def resolve_data_root(self) -> pathlib.Path | None:
        """Reuse the project's own resolution chain; absence is not fatal here."""
        if self.data_root is not None:
            path = pathlib.Path(self.data_root).expanduser()
            return path if path.is_dir() else None
        try:
            from fsgrade.config import resolve_data_root

            return resolve_data_root(None, None).path
        except Exception:  # noqa: BLE001 - the demo works fine without a dataset
            return None

    @property
    def is_public_bind(self) -> bool:
        return self.host not in ("127.0.0.1", "localhost", "::1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_root": str(self.results_root),
            "data_root": str(self.data_root) if self.data_root else None,
            "cache_root": str(self.cache_root),
            "device": self.resolve_device(),
            "host": self.host,
            "port": self.port,
            "clip_encoder": self.clip_encoder,
            "offline": self.offline,
            "public_bind": self.is_public_bind,
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "DemoSettings":
        def _path(key: str) -> pathlib.Path | None:
            raw = os.environ.get(key)
            return pathlib.Path(raw).expanduser() if raw else None

        return cls(
            results_root=_path("FSGRADE_RESULTS_ROOT") or pathlib.Path("results"),
            data_root=_path("FRUITVISION_DATA_ROOT"),
            cache_root=_path("FSGRADE_DEMO_CACHE") or _CACHE_ROOT,
            device=os.environ.get("FSGRADE_DEMO_DEVICE", "auto"),
            host=os.environ.get("FSGRADE_DEMO_HOST", "127.0.0.1"),
            port=int(os.environ.get("FSGRADE_DEMO_PORT", "8000")),
        )
