# Runbook — regenerating the thesis results

Everything you need to re-run, in order, with what to check at each step.

## Read this first: do not re-run the notebook

`notebooks/main_experiment.ipynb` is **not** the producer any more, and re-running
it would reintroduce the problem it caused:

- No code in it ever wrote `results/*.json`. Those files came from gitignored
  helper scripts that were deleted, so nothing in the repo could regenerate them.
- Its outputs were injected rather than executed — `execution_count` is 12, 13
  and 14 **twice each**, which a linear run cannot produce.
- It used the global `random` module, so numbers depended on cell execution
  order. That is why the same model appears at 85.0, 85.4, 85.6 and 85.7.

`scripts/reproduce_thesis.py` replaces it. One command regenerates every metric,
figure and checkpoint into a timestamped directory. Treat the notebook as a
historical artifact — or delete it from the submission.

---

## 0. Setup (5 min)

```bash
git pull                      # or copy the repo across
pip install -r requirements.txt
export FRUITVISION_ROOT=/path/to/FruitVision      # Windows: set FRUITVISION_ROOT=...
python scripts/reproduce_thesis.py --verify-data
```

**Expect:** per-species counts and `TOTAL 6978`, plus a manifest SHA-256.
**If the total is not 6978**, either the wrong tree is mounted or the thesis
dataset section needs updating. Do not continue until this matches.

## 1. Smoke run (~15 min on GPU)

```bash
python scripts/reproduce_thesis.py --seed 42 --smoke --skip-expensive
```

Proves every code path executes before you commit hours to it. I already ran this
end to end on a synthetic dataset and it completed clean in 746 s on CPU, so the
wiring is known good — this just confirms it on your machine and data.

**Expect:** `DONE in ...s -> results/run_smoke_<ts>_seed42`, with `metrics.json`,
`env.json`, `summary.csv`, `model_bundle.pt` and four PNGs in `figures/`.
Smoke runs are gitignored and ignored by `--aggregate`.

## 2. Baseline run, three seeds (the long one)

```bash
for s in 42 1337 2024; do python scripts/reproduce_thesis.py --seed $s; done
python scripts/reproduce_thesis.py --aggregate
```

This is the run the thesis quotes. Three seeds because every `±` you currently
report describes resampling of **one** trained model and says nothing about
training variance.

**Expect:** `results/seed_aggregate.json` with `mean ± sd over 3 seeds`.

**The numbers will move from your current ones.** That is correct, not a
regression — see §5.

## 3. Variant runs (choose the configuration)

```bash
python scripts/reproduce_thesis.py --seed 42 --freeze-bn-stats
python scripts/reproduce_thesis.py --seed 42 --val-protocol loso
python scripts/reproduce_thesis.py --seed 42 --val-protocol loso --freeze-bn-stats
```

| Flag | What it addresses |
|---|---|
| `--freeze-bn-stats` | Episodes are 32–40 images; BatchNorm estimated from that is far worse than ImageNet running statistics. Measured coupling drops 1.2e-01 → 6.7e-08. **Most likely accuracy win.** |
| `--val-protocol loso` | Model selection currently maximises *seen*-species accuracy (~0.94) while the thesis is about *unseen*-species accuracy (~0.85). LOSO validates on a training species held out entirely. |
| `--norm-layer layernorm` | Batch-size independent; fixes the degenerate 1-shot case (support batch of 2). |
| `--freeze-mode legacy_substring` | Reproduces the old over-broad freezing, for the ablation table. |

> ### Decide on validation, not on test
> Mango and orange are your held-out species. If you run all variants and report
> whichever scores best on them, the headline stops being a generalization claim
> and becomes a tuned number — and that is the one thing that would genuinely
> sink the thesis. Pick the configuration using **LOSO validation accuracy**, fix
> it, evaluate on mango/orange once, and report the rest as an ablation table.

## 4. Demo

```bash
python scripts/demo_app.py --selfcheck     # verify, no browser
python scripts/demo_app.py                 # serve on :8000
```

Draws a **live** episode on a held-out species and renders every support and
query image with prediction, confidence and correctness. Self-contained — no
external requests. Verified working against a real bundle.

Note: `fsgrade/demo` cannot load this model. Its loader rebuilds an *fsgrade*
architecture and calls `load_state_dict(..., strict=True)`, which a `src/`
checkpoint fails by construction. That is why the thesis model has its own demo.

## 5. What changed, and why numbers will move

| # | Change | Effect |
|---|---|---|
| 1 | **Layer freezing fixed** | The old substring match on `("layer1","conv1","bn1")` also matched `layer2.0.conv1`, `layer3.0.bn1`, … — the first conv and norm of *every block in every stage*. It froze **4,648,448 params (40.7%)** beyond the stem and layer1. Trainable was 6,765,569 — exactly your notebook's reported figure. Now 11,414,017. **This is the change most likely to move results.** |
| 2 | **Wilcoxon two-sided** | Was `alternative="greater"`, so baselines that *beat* you returned p = 1.00 and printed "No" — reading as "no difference" when the truth was "significantly worse". Now reports `ours better` / `ours WORSE` / `ns`. |
| 3 | **Seeded episode RNG** | Sampling used the global `random` module. Results depended on cell execution order. |
| 4 | **Support/query leak closed** | The fallback padded short query sets via `random.choices(all_images, ...)` — with replacement, from a list *including the support images*. Now raises. |
| 5 | **CI semantics separated** | Headline `±` was over **5 trial means**; every baseline `±` was over **600 episodes** — different quantities, identical notation. Now `ci_95_episode` and `ci_95_trial`, each with explicit `n`. Quote the episode-level one. |
| 6 | **Transfer controls added** | *Fine-tuned + Nearest Centroid* (Baseline++/SimpleShot) and *Supervised transfer (zero-shot)*. The second decides whether few-shot is even **necessary**. Both are paired into the significance tests. |
| 7 | **Hue jitter 0.05 → 0.02** | On a fresh/rotten task colour *is* the label. `configs/base.yaml:41-43` already said so; `src/` was never wired to it. |
| 8 | **Non-transductive forward** | Support and query embedded separately. **Affected training only** — under `model.eval()` the measured difference is exactly `0.00e+00`, so it did not change your reported test numbers. Do not overclaim it as a leak fix. |
| 9 | **t-SNE version fix** | `TSNE(max_iter=...)` only exists in scikit-learn ≥1.5. On 1.4.x it is a hard `TypeError` — this crashed the smoke run at the final step. Now detects the installed signature. |
| 10 | **Paths made portable** | `config.py` hardcoded `~/Desktop/Amr Samir` (nonexistent); `RESULTS_DIR="./results"` resolved relative to `notebooks/`, which is why every figure lookup printed `[Missing]` and `Imgs/` was filled by hand. |

## 6. Checklist before you write

- [ ] `--verify-data` reports 6,978 images
- [ ] Three seeds complete; `seed_aggregate.json` exists
- [ ] Every number in the thesis maps to a key in a `metrics.json`
- [ ] `pytest tests/` → 192 passed
- [ ] `python scripts/demo_app.py --selfcheck` → OK
- [ ] `RESULTS.md` regenerated; its ⚠ banner removed
- [ ] `RESULTS.md` states Siamese and Matching outperform the method, with
      two-sided p-values
- [ ] Transfer controls reported — **including if they win**
- [ ] EfficientNet-B0 (92.3%) and banana-only (90.9%) reported, not omitted
- [ ] Lead with the 10-fold CV mean, not the fixed split (which was 2nd best of 10)
- [ ] Notebook re-executed top to bottom, or removed from the deliverable

## 7. Files

**New**
- `scripts/reproduce_thesis.py` — the sole producer of results
- `scripts/demo_app.py` — live demo on the thesis model
- `REVIEW_FINDINGS.md` — 30 findings, severity-ranked, with evidence
- `NOVELTY_ASSESSMENT.md` — literature verdict, citations, dataset provenance
- `RUNBOOK.md` — this file

**Modified**
- `config.py` — portable paths, `HUE_JITTER`, `BATCH_SIZE` removed, weight-decay note
- `src/models.py` — freezing fix, `freeze_mode`, `norm_layer`, `freeze_bn_stats`,
  non-transductive forward
- `src/dataset.py` — seeded RNG, leak closed
- `src/experiments.py` — two-sided Wilcoxon, CI separation, transfer controls
- `src/transforms.py` — hue from config
- `src/visualization.py` — t-SNE compat, matrix+report from one prediction array
- `README.md`, `RESULTS.md`, `.gitignore`

**Still yours**
- The thesis LaTeX — you said you are rewriting it. Numbers must come from
  `metrics.json`, and `NOVELTY_ASSESSMENT.md` §2–3 lists the citations to add.
- The two paper drafts in `~/Downloads/Cross_*` — they may carry a third set of
  numbers and need the same reconciliation.
