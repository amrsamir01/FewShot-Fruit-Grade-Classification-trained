# Interactive demo

A live demonstration of cross-species few-shot adaptation, built for the viva.

> Pick a species the model has never seen. Give it *K* labelled examples. Watch
> the prototypes form. Grade queries — with the **zero-shot baseline running
> side-by-side**, so the room sees exactly what those *K* images bought.

---

## Install and run

```bash
pip install -r requirements-demo.txt          # fastapi, uvicorn, multipart, open_clip
python -m fsgrade.demo.prefetch               # ONCE, while online: caches CLIP weights
python -m fsgrade.demo --selfcheck            # reports exactly what will work
python -m fsgrade.demo --data-root D:/Datasets/FruitVision
```

The last command opens <http://localhost:8000>.

Useful flags:

| Flag | Effect |
|---|---|
| `--data-root PATH` | Enables one-click sampling from the local dataset |
| `--results-root PATH` | Where to look for trained runs (default `results`) |
| `--device cpu\|cuda\|auto` | CPU is fine — ResNet-18 grades an episode in well under a second |
| `--host 0.0.0.0` | Share on the LAN. **No authentication** — trusted networks only |
| `--no-browser` | Don't auto-open a tab |
| `--selfcheck` | Verify and exit; use this before the viva |

## What runs without any training

**SAP needs no checkpoint.** Its encoder is frozen CLIP and the only learned
quantity is κ, a single scalar. So the headline story works before a single
experiment has finished:

| Arm | Needs a checkpoint? | What it shows |
|---|---|---|
| `sap` | no | the proposed method, K-shot |
| `clip_text_zeroshot` | no | language only — **zero** images of the unseen species |
| `nc_pixel`, `chance` | no | the training-free floor |
| `zeroshot_supervised` | yes | **the pivotal control (E1)** |
| `ncc_supervised` | yes | same encoder + K-shot centroid |
| `ours`, `protonet`, `siamese`, `matching` | yes | the episodic family |

Arms whose checkpoint is missing are listed but disabled, with the reason shown.
Nothing crashes because a run has not happened yet.

## The three panels

**1 · Support set.** Choose a species — or type any fruit name, since the text
branch is not limited to the dataset. Drop images onto the *fresh* and *rotten*
trays, or click **Sample 5 + 5**. Support and query draws are kept disjoint, so
you are never grading an image the model was just shown.

**2 · Decision plane.** Points animate into place as the support set grows, so
the prototypes visibly *form* rather than appear.

**3 · Queries and verdicts.** Drop or paste query images, hit **Grade queries**,
and both arms answer the same question about the same images. The bottom row
states what the support set bought, in points.

## The decision plane is exact

The plot is **not** t-SNE. It is a linear projection onto the plane spanned by
the decision axis `e1 = normalize(p_rotten − p_fresh)` and the leading
orthogonal direction, with the origin at the prototype midpoint. Consequences:

- the boundary is **exactly** the vertical line `u = 0`;
- both prototypes land at exactly `(∓‖w‖/2, 0)`, symmetric about it;
- for a linear head and for scaled-cosine methods, the logit margin is exactly
  proportional to `u`.

Two honesty readouts sit above the plot:

| Readout | Meaning |
|---|---|
| **boundary agreement** | fraction of queries whose prediction matches the side of the drawn line. **Exactly 100%** for every method here. |
| **margin fidelity** | correlation between the true logit margin and in-plane distance. `1.000` for cosine and linear heads. Slightly below for ProtoNet and the pixel baseline, because they score with `−d` rather than `−d²` — monotone in `u`, not linear. |
| **plane captures** | share of the cloud's spread the plane holds. The rest is real structure the projection cannot show, and it is stated rather than hidden. |

If an examiner asks "is that just a t-SNE picture?", the answer is that a linear
projection preserves every linear decision function exactly, has units, has a
fixed frame, and puts the boundary where the model actually puts it.

## The κ dial

κ controls the hand-over from text to vision: `α_K = K / (K + κ)`. Drag it and
the prototypes move live — it only re-blends, so there is no re-embedding cost.

κ is **read, never fitted**. Calibrating it on the demo species would leak
target information and invalidate the cross-species claim. The provenance chain
is: a run's calibrated value → its calibration record → the default of 5.0,
labelled *"default, not calibrated"*. Dragging away from the calibrated value
flips the badge to *"exploring — κ ≠ calibrated"*.

## Offline guarantees

Everything that could reach the network, and how it is stopped:

| Path | Elimination |
|---|---|
| torchvision ImageNet download on model rebuild | `pretrained=False` is hardcoded in `checkpoints.py` and asserted in a test |
| CLIP weight download | `prefetch.py` caches them; `HF_HUB_OFFLINE=1`, `HF_HOME`, `TORCH_HOME` are set in `settings.py` **before** `open_clip` is imported |
| Swagger UI pulling from a CDN | `docs_url=None` (the OpenAPI JSON is still served) |
| Web fonts, CDN scripts, favicon fetch | zero external references; system font stack; favicon as a `data:` URI |
| HF telemetry | `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1` |

Three tests enforce this: one blocks all non-loopback sockets and asserts the
demo still starts, samples, predicts and projects; one greps every static asset
for external URLs; one asserts the offline environment is pinned at import.

## Sharing on the LAN

```bash
python -m fsgrade.demo --host 0.0.0.0 --data-root D:/Datasets/FruitVision
```

Then open `http://<your-ip>:8000` from another device on the same network. The
app has **no authentication** and the startup banner says so — use it on a
trusted network, and prefer loopback in the viva itself.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "CLIP unavailable" | `pip install open_clip_torch`, then `python -m fsgrade.demo.prefetch` while online |
| SAP arm disabled offline | Weights were never cached — run `prefetch` on a connected machine, then copy `.cache/` across |
| "no trained checkpoint found" | Expected until the experiments run. Use `sap` / `clip_text_zeroshot` meanwhile |
| Sampling buttons disabled | No `--data-root`. Uploads still work |
| Uploads fail with a 500 | `python-multipart` missing — `pip install python-multipart` |
| Checkpoint "does not match the rebuilt architecture" | The run's `config.yaml` does not describe those weights. Load the run directory that produced the checkpoint |
