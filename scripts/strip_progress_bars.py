"""
Collapse tqdm progress-bar redraws in an executed notebook.

tqdm repaints its bar by emitting a carriage return and rewriting the line, and
ipykernel records every repaint as its own stream output. A long run therefore
leaves hundreds of thousands of near-identical output objects in the .ipynb even
though the terminal only ever displayed the last one. On this project's
main_experiment.ipynb that was 237,729 stream outputs (126,041 in the 10-split
cross-validation cell alone), inflating the file to 48 MB against just 0.4 MB of
actual figures.

Two passes, both terminal-faithful:

  1. Merge runs of consecutive stream outputs that share a stream name, so a
     bar's repaints end up in one string rather than one object each.
  2. Within each line, keep only the segment after the final carriage return -
     exactly what the terminal showed.

tqdm writes to stderr and the experiment code prints results to stdout, and the
two are merged separately, so no printed table, accuracy or heading is touched.
Lines containing no carriage return are never modified.

    python scripts/strip_progress_bars.py notebooks/main_experiment.ipynb
"""

from __future__ import annotations

import json
import pathlib
import sys


def collapse(text: str) -> list[str]:
    lines = []
    for line in text.split("\n"):
        shown = line.rsplit("\r", 1)[-1] if "\r" in line else line
        lines.append(shown)
    # tqdm leaves blank filler lines behind when it clears a finished bar.
    while lines and lines[-1].strip() == "":
        lines.pop()
    return [ln + "\n" for ln in lines]


def squash(outputs: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for out in outputs:
        if (out.get("output_type") == "stream"
                and merged
                and merged[-1].get("output_type") == "stream"
                and merged[-1].get("name") == out.get("name")):
            merged[-1]["text"].append("".join(out.get("text", [])))
        elif out.get("output_type") == "stream":
            merged.append({"output_type": "stream",
                           "name": out.get("name", "stdout"),
                           "text": ["".join(out.get("text", []))]})
        else:
            merged.append(out)

    for out in merged:
        if out.get("output_type") == "stream":
            out["text"] = collapse("".join(out["text"]))
    return [o for o in merged
            if o.get("output_type") != "stream" or "".join(o["text"]).strip()]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = pathlib.Path(sys.argv[1])
    before = path.stat().st_size
    nb = json.loads(path.read_text(encoding="utf-8"))

    n_before = n_after = 0
    for cell in nb["cells"]:
        outs = cell.get("outputs")
        if not outs:
            continue
        n_before += len(outs)
        cell["outputs"] = squash(outs)
        n_after += len(cell["outputs"])

    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    after = path.stat().st_size
    print(f"{path.name}: {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({100 * (1 - after / before):.1f}% smaller)")
    print(f"outputs: {n_before:,} -> {n_after:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
