#!/usr/bin/env python
"""
Live few-shot demo for the thesis model.

Runs a real 5-shot episode on a held-out species: sample a support set, form
prototypes, classify queries, and show what the model actually decided. Nothing
is precomputed — every reload draws a fresh episode.

Why this is separate from fsgrade/demo
--------------------------------------
The fsgrade demo cannot load this model. Its discovery scans a two-level
results/<experiment>/<run>/ tree, and load_for_inference() rebuilds an *fsgrade*
architecture then calls load_state_dict(..., strict=True). A src/
PrototypicalNetwork has different parameter names and shapes, so that fails by
construction. Bridging it would mean putting a src/-specific adapter inside
fsgrade, which is out of scope: fsgrade is the engineering deliverable, src/ is
the thesis pipeline. So the thesis model gets its own demo.

Usage
-----
    export FRUITVISION_ROOT=/path/to/FruitVision
    python scripts/demo_app.py --bundle results/run_<ts>_seed42/model_bundle.pt
    python scripts/demo_app.py --selfcheck     # no browser, verifies wiring

The page is fully self-contained: inline CSS, inline JS, images embedded as
data: URIs. No external requests.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import webbrowser
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config  # noqa: E402
from src.dataset import FruitQualityDataset  # noqa: E402
from src.models import PrototypicalNetwork  # noqa: E402
from src.transforms import build_transforms  # noqa: E402


# ====================================================================== #
#  Model + data
# ====================================================================== #

def latest_bundle() -> Path | None:
    """Newest model_bundle.pt under results/, ignoring smoke runs."""
    found = sorted(
        (p for p in (REPO_ROOT / "results").glob("run_*/model_bundle.pt")
         if "smoke" not in p.parent.name),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return found[0] if found else None


def load_model(bundle_path: Path, device: torch.device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    arch = bundle.get("arch", {})
    model = PrototypicalNetwork(
        backbone=arch.get("backbone", "resnet18"),
        embedding_dim=arch.get("embedding_dim", 256),
        pretrained=False,
        dropout_rate=arch.get("dropout_rate", 0.4),
        temperature=bundle.get("temperature", 0.5),
        freeze_mode=arch.get("freeze_mode", "stem_layer1"),
        norm_layer=arch.get("norm_layer", "batchnorm"),
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.eval()
    return model, bundle


def to_data_uri(path: str, size: int = 128) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_episode(model, dataset, bundle, device, species: str, n_shot: int):
    """One real episode. Returns everything the page needs to render."""
    n_query = bundle.get("n_query", 15)
    classes = bundle.get("classes", ["fresh", "rotten"])

    # Paths first, so the page can show the actual images used.
    rng = dataset.rng
    support_paths, query_paths, query_truth = [], [], []
    for class_idx, quality in enumerate(classes):
        pool = dataset.data[species][quality]
        picked = rng.sample(pool, n_shot + n_query)
        support_paths.extend((p, class_idx) for p in picked[:n_shot])
        for p in picked[n_shot:]:
            query_paths.append(p)
            query_truth.append(class_idx)

    from PIL import Image
    tf = dataset.transform

    def batch(paths):
        return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)

    s_imgs = batch([p for p, _ in support_paths])
    s_lbls = torch.tensor([c for _, c in support_paths], device=device)
    q_imgs = batch(query_paths)
    q_true = torch.tensor(query_truth, device=device)

    with torch.no_grad():
        logits, q_emb, s_emb, protos = model(s_imgs, s_lbls, q_imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)

    correct = (preds == q_true)
    queries = [
        {
            "img": to_data_uri(path, 96),
            "true": classes[t],
            "pred": classes[p.item()],
            "confidence": round(float(probs[i, p].item()), 3),
            "correct": bool(correct[i].item()),
        }
        for i, (path, t, p) in enumerate(zip(query_paths, query_truth, preds))
    ]

    return {
        "species": species,
        "n_shot": n_shot,
        "accuracy": round(float(correct.float().mean().item()), 4),
        "n_query": len(queries),
        "temperature": round(float(model.temperature.detach().cpu()), 4),
        "prototype_separation": round(
            float(torch.cdist(protos[:1], protos[1:]).item()), 4),
        "support": [
            {"img": to_data_uri(p, 96), "label": classes[c]} for p, c in support_paths
        ],
        "queries": queries,
        "per_class": {
            classes[c]: round(
                float(correct[q_true == c].float().mean().item()), 4)
            for c in range(len(classes))
        },
    }


# ====================================================================== #
#  Page
# ====================================================================== #

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Few-Shot Fruit Quality Grading</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--ok:#2d7a4f;--bad:#b3261e;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#f0eee9;--mut:#a3a09a;--line:#332f28;--ok:#6fcf97;--bad:#ef5350;--card:#1f1d17}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.5rem;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .25rem}
.sub{color:var(--mut);margin:0 0 1.5rem;font-size:.9rem}
.ctl{display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin-bottom:1.5rem}
select,button{font:inherit;padding:.5rem .9rem;border:1px solid var(--line);
 border-radius:8px;background:var(--card);color:var(--fg)}
button{cursor:pointer;font-weight:600}
button:hover{border-color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
 gap:.75rem;margin-bottom:1.5rem}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.85rem 1rem}
.stat .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.stat .v{font-size:1.45rem;font-weight:650;margin-top:.15rem}
h2{font-size:.95rem;margin:1.5rem 0 .6rem;text-transform:uppercase;
 letter-spacing:.05em;color:var(--mut)}
.grid{display:flex;flex-wrap:wrap;gap:.6rem}
.cell{background:var(--card);border:1px solid var(--line);border-radius:9px;
 padding:.4rem;width:104px;text-align:center}
.cell img{width:100%;border-radius:5px;display:block}
.cell .lab{font-size:.72rem;margin-top:.3rem;line-height:1.3}
.cell.ok{border-color:var(--ok)} .cell.bad{border-color:var(--bad);border-width:2px}
.ok-t{color:var(--ok)} .bad-t{color:var(--bad)}
.note{color:var(--mut);font-size:.83rem;margin-top:1.75rem;
 border-top:1px solid var(--line);padding-top:1rem}
</style></head><body><div class="wrap">
<h1>Few-Shot Fruit Quality Grading</h1>
<p class="sub">Live episode on a species the model never trained on. Support set
defines the classes; queries are classified by distance to the resulting prototypes.</p>
<div class="ctl">
  <label>Species <select id="sp"></select></label>
  <label>Shots <select id="k"><option>1</option><option selected>5</option><option>10</option></select></label>
  <button onclick="go()">New episode</button>
</div>
<div id="out"></div>
<p class="note" id="note"></p>
</div><script>
let META=null;
async function boot(){
  META=await (await fetch('/api/meta')).json();
  const sp=document.getElementById('sp');
  sp.innerHTML=META.test_fruits.map(f=>`<option>${f}</option>`).join('');
  document.getElementById('note').textContent=
    `Model: ${META.backbone}, ${META.norm_layer}, freeze=${META.freeze_mode}. `+
    `Trained on ${META.train_fruits.join(', ')} — never on ${META.test_fruits.join(' or ')}. `+
    `Bundle: ${META.bundle}`;
  go();
}
async function go(){
  const out=document.getElementById('out');
  out.innerHTML='<p class="sub">Running episode…</p>';
  const sp=document.getElementById('sp').value, k=document.getElementById('k').value;
  const r=await fetch(`/api/episode?species=${sp}&n_shot=${k}`);
  if(!r.ok){out.innerHTML=`<p class="bad-t">${(await r.json()).detail}</p>`;return;}
  const d=await r.json();
  const pc=Object.entries(d.per_class)
    .map(([c,v])=>`<div class="stat"><div class="k">${c} recall</div><div class="v">${(v*100).toFixed(1)}%</div></div>`).join('');
  out.innerHTML=`
  <div class="stats">
    <div class="stat"><div class="k">Episode accuracy</div><div class="v">${(d.accuracy*100).toFixed(1)}%</div></div>
    ${pc}
    <div class="stat"><div class="k">Prototype gap</div><div class="v">${d.prototype_separation}</div></div>
    <div class="stat"><div class="k">Temperature</div><div class="v">${d.temperature}</div></div>
  </div>
  <h2>Support set — ${d.n_shot} per class, all the model is given</h2>
  <div class="grid">${d.support.map(s=>
    `<div class="cell"><img src="${s.img}"><div class="lab">${s.label}</div></div>`).join('')}</div>
  <h2>Queries — ${d.n_query} classified</h2>
  <div class="grid">${d.queries.map(q=>
    `<div class="cell ${q.correct?'ok':'bad'}"><img src="${q.img}">
     <div class="lab"><span class="${q.correct?'ok-t':'bad-t'}">${q.pred}</span>
     ${q.correct?'':`<br><s>${q.true}</s>`}<br>${(q.confidence*100).toFixed(0)}%</div></div>`).join('')}</div>`;
}
boot();
</script></body></html>"""


# ====================================================================== #
#  Server
# ====================================================================== #

def build_app(model, bundle, datasets, device):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Few-Shot Fruit Quality Grading", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/meta")
    def meta():
        arch = bundle.get("arch", {})
        return JSONResponse({
            "train_fruits": bundle.get("train_fruits", []),
            "test_fruits": [f for f in bundle.get("test_fruits", []) if f in datasets],
            "backbone": arch.get("backbone", "resnet18"),
            "norm_layer": arch.get("norm_layer", "batchnorm"),
            "freeze_mode": arch.get("freeze_mode", "stem_layer1"),
            "bundle": bundle.get("_path", ""),
        })

    @app.get("/api/episode")
    def episode(species: str, n_shot: int = 5):
        if species not in datasets:
            raise HTTPException(404, f"No data loaded for species '{species}'")
        try:
            return JSONResponse(
                run_episode(model, datasets[species], bundle, device, species, n_shot))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default=None,
                   help="Path to model_bundle.pt. Defaults to the newest under results/.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--selfcheck", action="store_true",
                   help="Load model and run one episode per species, then exit.")
    args = p.parse_args()

    # resolve() so a relative --bundle still compares against REPO_ROOT below.
    bundle_path = Path(args.bundle).resolve() if args.bundle else latest_bundle()
    if not bundle_path or not bundle_path.exists():
        print("No model bundle found. Train one first:")
        print("  python scripts/reproduce_thesis.py --seed 42")
        return 1

    config = Config()
    if not Path(config.DATA_ROOT).is_dir():
        print(f"Dataset not found at {config.DATA_ROOT}. Set FRUITVISION_ROOT.")
        return 1

    device = torch.device(args.device)
    print(f"Loading {bundle_path}")
    model, bundle = load_model(bundle_path, device)
    try:
        bundle["_path"] = str(bundle_path.relative_to(REPO_ROOT))
    except ValueError:
        # Bundle lives outside the repo (copied off the GPU box, say).
        bundle["_path"] = str(bundle_path)

    tf = build_transforms(config)
    datasets = {}
    for species in bundle.get("test_fruits", config.TEST_FRUITS):
        ds = FruitQualityDataset(config.DATA_ROOT, [species], config.CLASSES,
                                 transform=tf["eval"], split="all", seed=0)
        if all(ds.data[species][q] for q in config.CLASSES):
            datasets[species] = ds
    if not datasets:
        print("No held-out species found in the dataset.")
        return 1

    if args.selfcheck:
        ok = True
        for species in datasets:
            try:
                r = run_episode(model, datasets[species], bundle, device, species, 5)
                print(f"  {species:<8} accuracy {r['accuracy']*100:5.1f}%  "
                      f"({r['n_query']} queries)")
            except Exception as exc:  # noqa: BLE001
                print(f"  {species:<8} FAILED: {type(exc).__name__}: {exc}")
                ok = False
        print("selfcheck:", "OK" if ok else "FAILED")
        return 0 if ok else 1

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {url}  (species: {', '.join(datasets)})")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(build_app(model, bundle, datasets, device),
                host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
