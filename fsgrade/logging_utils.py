"""
Logging that survives headless runs.

Three of the eight advertised experiments in the original notebook produced no
recorded numbers at all: their stdout was truncated by Jupyter's IOPub rate
limit mid-run. Nothing was written to disk, so those results are simply gone.

Every log line here goes to a file as well as the console, and progress
reporting has a ``plain`` mode that emits one line per N episodes rather than a
continuously-repainting bar -- safe for ``nohup``, CI, and notebooks alike.
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from typing import Any, Iterable, Iterator, Sequence

_CONFIGURED: dict[str, logging.Logger] = {}

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "fsgrade",
    *,
    log_file: str | pathlib.Path | None = None,
    level: str | int = "INFO",
    file_level: str | int = "DEBUG",
) -> logging.Logger:
    """Return a logger writing to stdout at ``level`` and to ``log_file`` at ``file_level``.

    Per-episode chatter is logged at DEBUG so it lands in the file but never
    floods the console.
    """
    key = f"{name}:{log_file}"
    if key in _CONFIGURED:
        return _CONFIGURED[key]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(stream)

    if log_file is not None:
        path = pathlib.Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)

    _CONFIGURED[key] = logger
    return logger


def progress(
    iterable: Iterable[Any],
    *,
    desc: str = "",
    total: int | None = None,
    mode: str = "plain",
    logger: logging.Logger | None = None,
    every: int = 100,
) -> Iterator[Any]:
    """Iterate with progress reporting.

    mode='tqdm'  -- a normal progress bar (interactive use)
    mode='plain' -- one log line every ``every`` items (headless use; the mode
                    that would have preserved the three lost experiments)
    mode='none'  -- silent
    """
    if mode == "tqdm":
        try:
            from tqdm.auto import tqdm

            yield from tqdm(iterable, desc=desc, total=total, leave=False)
            return
        except ImportError:
            mode = "plain"

    if mode == "none":
        yield from iterable
        return

    log = logger or get_logger()
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    started = time.time()
    count = 0
    for item in iterable:
        yield item
        count += 1
        if count % every == 0:
            elapsed = time.time() - started
            rate = count / elapsed if elapsed > 0 else 0.0
            if total:
                eta = (total - count) / rate if rate > 0 else float("nan")
                log.info(
                    "%s %d/%d (%.1f%%) | %.1f it/s | ETA %.0fs",
                    desc, count, total, 100.0 * count / total, rate, eta,
                )
            else:
                log.info("%s %d | %.1f it/s", desc, count, rate)

    elapsed = time.time() - started
    log.info("%s done: %d items in %.1fs", desc, count, elapsed)


def log_table(
    logger: logging.Logger,
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    *,
    title: str | None = None,
) -> None:
    """Log a fixed-width table -- readable in a plain-text log file."""
    cols = [str(h) for h in headers]
    widths = [len(c) for c in cols]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    if title:
        logger.info(title)
    sep = "-+-".join("-" * w for w in widths)
    logger.info("  " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    logger.info("  " + sep)
    for row in str_rows:
        logger.info("  " + " | ".join(c.ljust(w) for c, w in zip(row, widths)))
