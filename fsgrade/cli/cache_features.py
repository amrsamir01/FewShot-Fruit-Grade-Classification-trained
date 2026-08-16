"""
Pre-extract frozen foundation-model features.

One GPU pass over the ~7k images per encoder. Afterwards every frozen method
(nearest-centroid, linear probe, SAP) is seconds of CPU work, which is what
makes the full ladder affordable.

    python -m fsgrade.cli.cache_features --encoders dinov2_vits14 clip_vitb16
"""

from __future__ import annotations

from fsgrade.cli.common import base_parser, build_index, setup
from fsgrade.config import get_in


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Cache frozen foundation-model features.")
    parser.add_argument("--encoders", nargs="+",
                        default=["dinov2_vits14", "clip_vitb16"],
                        help="Encoder names to cache")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)

    ctx = setup(args)
    cfg, device, logger = ctx["cfg"], ctx["device"], ctx["logger"]
    index = build_index(cfg)

    from fsgrade.models.foundation import FoundationUnavailable, get_or_build_cache

    failures: dict[str, str] = {}
    for name in args.encoders:
        try:
            logger.info("caching %s on %s ...", name, device)
            cache = get_or_build_cache(
                name, index,
                cache_root=get_in(cfg, "paths.cache_root", "./.cache"),
                device=device,
                batch_size=args.batch_size or int(get_in(cfg, "eval.feature_batch_size", 64)),
                num_workers=int(get_in(cfg, "num_workers", 0)),
                logger=logger,
            )
            paths, feats = cache.load()
            logger.info("  %s -> %d x %d features at %s",
                        name, feats.shape[0], feats.shape[1], cache.directory)
        except (FoundationUnavailable, ImportError) as exc:
            failures[name] = str(exc)
            logger.warning("  %s unavailable: %s", name, exc)

    if failures:
        logger.warning("%d encoder(s) skipped. Install with: "
                       "pip install timm open_clip_torch", len(failures))
        return 1 if len(failures) == len(args.encoders) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
