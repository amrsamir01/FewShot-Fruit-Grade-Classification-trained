"""Shared CLI plumbing: argument parsing, device selection, context assembly."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import torch

from fsgrade.config import build_config, get_in
from fsgrade.logging_utils import get_logger
from fsgrade.seeding import set_seed


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/experiment/main_loso.yaml",
                   help="Path to an experiment YAML config")
    p.add_argument("--data-root", default=None,
                   help="Dataset root (overrides config and environment)")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Dotted config overrides, e.g. model.freeze.stage=2")
    p.add_argument("--device", default=None, help="cpu | cuda | auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-level", default=None)
    return p


def resolve_device(cfg: dict[str, Any], override: str | None = None) -> torch.device:
    choice = override or get_in(cfg, "device", "auto")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def setup(args: argparse.Namespace, *, validate_data: bool = True) -> dict[str, Any]:
    """Build config, seed everything, and return a context dict."""
    cfg = build_config(
        args.config,
        overrides=args.overrides,
        data_root=args.data_root,
        validate_data=validate_data,
    )
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed_report = set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    device = resolve_device(cfg, args.device)
    logger = get_logger(
        "fsgrade", level=args.log_level or get_in(cfg, "logging.level", "INFO")
    )
    return {"cfg": cfg, "device": device, "logger": logger, "seed_report": seed_report}


def build_index(cfg: dict[str, Any]):
    from fsgrade.data.index import ImageIndex

    return ImageIndex.build(
        get_in(cfg, "paths.data_root"),
        get_in(cfg, "data.species", []),
        get_in(cfg, "data.classes", []),
        extensions=get_in(cfg, "data.extensions", (".jpg", ".jpeg", ".png", ".bmp")),
    )


def banks_dir(cfg: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(get_in(cfg, "paths.episodes_root", "./results/_episodes"))
    path.mkdir(parents=True, exist_ok=True)
    return path
