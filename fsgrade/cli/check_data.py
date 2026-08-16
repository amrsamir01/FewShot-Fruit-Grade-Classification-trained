"""
Step 0 of every run and of CI: validate the dataset before touching a GPU.

Exits non-zero with the exact missing directories. The original loader merely
printed ``WARNING Missing folder: ...``, carried on with an empty image list,
and then failed deep inside a training loop -- or worse, produced partially
populated episodes.

    python -m fsgrade.cli.check_data --config configs/experiment/main_loso.yaml
"""

from __future__ import annotations

import sys
from typing import Any

from fsgrade.cli.common import base_parser, build_index, setup
from fsgrade.config import get_in
from fsgrade.logging_utils import log_table


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Validate the dataset layout and report integrity statistics.")
    parser.add_argument("--duplicates", action="store_true",
                        help="Hash every image to detect exact duplicates (slower)")
    parser.add_argument("--corrupt", action="store_true",
                        help="Verify every image opens (slower)")
    parser.add_argument("--write-manifest", action="store_true",
                        help="Write dataset_manifest.csv + .meta.json")
    args = parser.parse_args(argv)

    try:
        ctx = setup(args, validate_data=True)
    except Exception as exc:
        print(f"\nDATASET CHECK FAILED\n\n{exc}\n", file=sys.stderr)
        return 2

    cfg, logger = ctx["cfg"], ctx["logger"]
    logger.info("data_root = %s  (resolved via %s)",
                get_in(cfg, "paths.data_root"), get_in(cfg, "paths.data_root_source"))

    index = build_index(cfg)
    counts = index.counts()
    classes = list(index.classes)

    rows = []
    for species in index.species:
        per = counts[species]
        rows.append([species, *[per[c] for c in classes], sum(per.values())])
    totals = [sum(counts[s][c] for s in index.species) for c in classes]
    rows.append(["TOTAL", *totals, sum(totals)])

    log_table(logger, rows, ["species", *classes, "total"],
              title=f"Dataset: {len(index)} images across {len(index.species)} species")
    logger.info("manifest hash: %s", index.manifest_hash)

    ok = True

    # Capacity check for the configured episode geometry.
    n_shot_max = int(get_in(cfg, "protocol.episodes.n_shot_max", 10))
    n_query = int(get_in(cfg, "protocol.episodes.n_query_per_class", 15))
    try:
        index.require_capacity(index.species, n_shot=n_shot_max, n_query_per_class=n_query)
        logger.info("capacity OK: every species/class supports %d support + %d query",
                    n_shot_max, n_query)
    except Exception as exc:
        logger.error("capacity check FAILED:\n%s", exc)
        ok = False

    if args.duplicates:
        logger.info("hashing %d images for duplicate detection...", len(index))
        dupes = index.find_duplicates()
        if dupes:
            ok = False
            logger.warning("found %d duplicate groups covering %d images",
                           len(dupes), sum(len(v) for v in dupes.values()))
            for h, paths in list(dupes.items())[:10]:
                logger.warning("  %s: %s", h[:12], paths)
            cross = {h: p for h, p in dupes.items()
                     if len({x.split("/")[1] for x in p}) > 1}
            if cross:
                logger.error("%d duplicate groups span BOTH quality classes -- these "
                             "would inflate every reported number", len(cross))
        else:
            logger.info("no exact duplicates found")

    if args.corrupt:
        logger.info("verifying %d images open correctly...", len(index))
        bad = index.find_corrupt()
        if bad:
            ok = False
            logger.error("%d unreadable images, e.g. %s", len(bad), bad[:3])
        else:
            logger.info("all images readable")

    if args.write_manifest:
        path = index.write_manifest(
            get_in(cfg, "paths.index_root", "./results/_index") + "/dataset_manifest.csv"
        )
        logger.info("wrote manifest -> %s", path)

    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
