"""
Launch the demo.

    python -m fsgrade.demo                       # localhost, opens a browser
    python -m fsgrade.demo --selfcheck           # verify without serving
    python -m fsgrade.demo --host 0.0.0.0        # share on the LAN

Importing ``fsgrade.demo.settings`` pins the model caches and disables outbound
Hugging Face calls *before* anything imports open_clip or timm.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import webbrowser

from fsgrade.demo.settings import DemoSettings


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fsgrade.demo", description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--results-root", default="results")
    p.add_argument("--data-root", default=None,
                   help="FruitVision root; enables sampling from the dataset")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--clip-encoder", default="clip_vitb16")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--selfcheck", action="store_true",
                   help="Check everything the demo needs, then exit")
    return p


def _settings(args: argparse.Namespace) -> DemoSettings:
    return DemoSettings(
        results_root=pathlib.Path(args.results_root),
        data_root=pathlib.Path(args.data_root) if args.data_root else None,
        device=args.device,
        host=args.host,
        port=args.port,
        clip_encoder=args.clip_encoder,
    )


def selfcheck(settings: DemoSettings) -> int:
    """Report exactly what will and will not work, and why."""
    ok = True
    print("=" * 66)
    print("  Cross-Species Quality Grading — demo self-check")
    print("=" * 66)

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        print(f"  [ok]   fastapi + uvicorn")
    except ImportError as exc:
        ok = False
        print(f"  [FAIL] web stack missing: {exc}")
        print("         pip install -r requirements-demo.txt")

    try:
        import multipart  # noqa: F401

        print("  [ok]   python-multipart (uploads)")
    except ImportError:
        ok = False
        print("  [FAIL] python-multipart missing -- uploads will fail")
        print("         pip install python-multipart")

    device = settings.resolve_device()
    print(f"  [ok]   device: {device}")

    data_root = settings.resolve_data_root()
    if data_root:
        print(f"  [ok]   dataset: {data_root}")
    else:
        print("  [warn] no dataset found -- uploads work, sampling is disabled")
        print("         pass --data-root /path/to/FruitVision")

    from fsgrade.demo.engine import InferenceEngine

    engine = InferenceEngine(settings.results_root, device=device,
                             clip_encoder=settings.clip_encoder)
    runs = engine.runs(refresh=True)
    print(f"  [{'ok' if runs else 'warn'}]   runs discovered: {len(runs)}")
    for run in runs[:5]:
        arms = sorted(m.name for m in run.available_methods())
        print(f"           {run.experiment}/{run.run_id}: {', '.join(arms) or 'none'}")

    caps = engine.capabilities()
    if caps["clip"]:
        print(f"  [ok]   CLIP available (kappa={caps['kappa']}, {caps['kappa_source']})")
    else:
        print(f"  [warn] CLIP unavailable: {caps['clip_error']}")
        print("         pip install open_clip_torch && python -m fsgrade.demo.prefetch")

    print(f"  [ok]   training-free arms: {', '.join(caps['training_free_arms'])}")
    if caps["checkpoint_arms"]:
        print(f"  [ok]   trained arms: {', '.join(caps['checkpoint_arms'])}")
    else:
        print("  [warn] no trained checkpoints -- the zero-shot supervised control "
              "(E1) is unavailable")

    static = pathlib.Path(__file__).parent / "static"
    external = _scan_external_refs(static)
    if external:
        ok = False
        print(f"  [FAIL] {len(external)} external reference(s) in static assets:")
        for line in external[:5]:
            print(f"           {line}")
    else:
        print("  [ok]   static assets make no external requests")

    print("=" * 66)
    print("  RESULT:", "READY" if ok else "PROBLEMS FOUND")
    if not caps["clip"] and not caps["checkpoint_arms"]:
        print("  Note: with neither CLIP nor checkpoints, only the pixel and chance")
        print("        arms will run. The demo will still start.")
    return 0 if ok else 1


def _scan_external_refs(static_dir: pathlib.Path) -> list[str]:
    """Any http(s) or protocol-relative URL in the frontend breaks offline use."""
    import re

    if not static_dir.is_dir():
        return []
    pattern = re.compile(r"""(?:https?:)?//[a-z0-9.\-]+\.[a-z]{2,}""", re.I)
    hits: list[str] = []
    for path in static_dir.rglob("*"):
        if path.suffix.lower() not in (".html", ".css", ".js"):
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            # Comments and the XML namespace in the inline favicon are inert.
            if stripped.startswith(("//", "*", "/*", "#")) or "www.w3.org" in line:
                continue
            for match in pattern.findall(line):
                if "w3.org" not in match:
                    hits.append(f"{path.relative_to(static_dir)}:{n}: {match}")
    return hits


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = _settings(args)

    if args.selfcheck:
        return selfcheck(settings)

    try:
        import uvicorn
    except ImportError:
        print("The demo needs fastapi and uvicorn:\n"
              "  pip install -r requirements-demo.txt", file=sys.stderr)
        return 2

    from fsgrade.demo.app import create_app

    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print("=" * 66)
    print("  Cross-Species Quality Grading — demo")
    print(f"  {url}")
    if settings.is_public_bind:
        print("  WARNING: bound to a public interface with NO authentication.")
        print("           Use only on a trusted network.")
    print("=" * 66)

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
