"""
Download the frozen foundation weights once, while you still have internet.

    python -m fsgrade.demo.prefetch                  # fetch, then verify offline
    python -m fsgrade.demo.prefetch --verify-only

Run this **before** travelling to the viva. Afterwards the demo runs with
``HF_HUB_OFFLINE=1`` and never touches the network; this script is the only
place that is allowed to.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ASSETS_FILE = "demo_assets.json"


def _fetch(encoders: list[str], cache_root: pathlib.Path) -> dict[str, str]:
    """Build each encoder once with the network enabled."""
    os.environ["FSGRADE_DEMO_ONLINE"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from fsgrade.demo.settings import enforce_offline

    enforce_offline(cache_root, offline=False)

    import torch

    from fsgrade.models.foundation import build_foundation_encoder

    report: dict[str, str] = {}
    for name in encoders:
        print(f"  fetching {name} ...", flush=True)
        try:
            enc = build_foundation_encoder(name, device=torch.device("cpu"))
            report[name] = f"ok (dim={enc.out_dim}, input={enc.input_size})"
            print(f"    {report[name]}")
        except Exception as exc:  # noqa: BLE001
            report[name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"    {report[name]}", file=sys.stderr)
    return report


def _verify_offline(encoders: list[str], cache_root: pathlib.Path) -> dict[str, str]:
    """Re-build every encoder in a subprocess with the network disabled.

    A separate process is essential: open_clip and timm read the offline
    environment variables at *import* time, so flipping them in-process after
    they are already loaded proves nothing.
    """
    script = (
        "import json,sys,torch;"
        "from fsgrade.models.foundation import build_foundation_encoder as b;"
        "out={};\n"
        "for n in sys.argv[1:]:\n"
        "    try:\n"
        "        e=b(n, device=torch.device('cpu')); out[n]='ok'\n"
        "    except Exception as ex:\n"
        "        out[n]=f'FAILED: {type(ex).__name__}: {ex}'\n"
        "print('___JSON___'+json.dumps(out))"
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(cache_root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
        "TORCH_HOME": str(cache_root / "torch"),
    }
    env.pop("FSGRADE_DEMO_ONLINE", None)

    proc = subprocess.run(
        [sys.executable, "-c", script, *encoders],
        capture_output=True, text=True, env=env, timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("___JSON___"):
            return json.loads(line[len("___JSON___"):])
    return {n: f"verification failed: {proc.stderr.strip()[:200]}" for n in encoders}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsgrade.demo.prefetch",
                                     description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["clip_vitb16"],
                        help="Encoders to cache (add dinov2_vits14 if you want it)")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    cache_root = pathlib.Path(
        args.cache_root or os.environ.get("FSGRADE_DEMO_CACHE") or (pathlib.Path.cwd() / ".cache")
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  Prefetching demo weights into {cache_root}")
    print("=" * 66)

    try:
        import open_clip  # noqa: F401
    except ImportError:
        print("open_clip_torch is not installed:\n"
              "  pip install open_clip_torch", file=sys.stderr)
        return 2

    fetched = {} if args.verify_only else _fetch(args.encoders, cache_root)

    print("\n  verifying with HF_HUB_OFFLINE=1 in a fresh process ...")
    verified = _verify_offline(args.encoders, cache_root)
    for name, status in verified.items():
        print(f"    {name}: {status}")

    payload = {
        "cache_root": str(cache_root),
        "encoders": args.encoders,
        "fetched": fetched,
        "verified_offline": verified,
    }
    out = cache_root / ASSETS_FILE
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")

    ok = all(v == "ok" for v in verified.values())
    print("=" * 66)
    print("  RESULT:", "READY FOR OFFLINE USE" if ok else "NOT FULLY CACHED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
