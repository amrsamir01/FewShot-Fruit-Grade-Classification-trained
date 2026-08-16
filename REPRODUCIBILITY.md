# Reproducibility

Everything needed to re-run this work on another machine.

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                     # makes `python -m fsgrade.cli.*` work anywhere
```

**Required for the foundation-model arms — not actually optional for this thesis:**

```bash
pip install timm open_clip_torch          # or: pip install -e ".[foundation]"
```

Without them, DINOv2/CLIP/SAP are **skipped with a recorded reason** in
`metrics.json` rather than crashing the sweep. That graceful degradation is a
trap here: the `env.json` from the existing smoke runs records `timm: null` and
`open_clip: null`, meaning **SAP — the proposed method — has never executed**, and
a full ladder run on that environment would quietly omit it.

Before scheduling any long run, confirm the arms are live:

```bash
python -c "import timm, open_clip; print(timm.__version__, open_clip.__version__)"
# then, after a real run:
#   metrics.json must show sap with n_episodes > 0 and no MethodUnavailable reason
```

The environment actually used for a run is captured in `env.json`: Python and
package versions, GPU name and memory, CUDA/cuDNN versions, git commit, a hash
of any uncommitted diff, and the determinism report.

## 2. Dataset

Expected layout:

```
<data_root>/<species>/<quality>/*.jpg      quality ∈ {fresh, rotten}
```

Resolution order (first hit wins, every step logged into `config.yaml`):

1. `--data-root <path>`
2. `$FRUITVISION_DATA_ROOT`
3. `configs/local.yaml` (git-ignored; copy from `local.yaml.example`)
4. `paths.data_root` in the config
5. a probed candidate list

Validate before anything else — this exits non-zero and names the exact missing
directories:

```bash
python -m fsgrade.cli.check_data --config configs/experiment/main_loso.yaml \
       --duplicates --write-manifest
```

**Verified reference counts** (image-extension filter, matching the recorded
notebook output exactly):

| species | fresh | rotten | total |
|---|---:|---:|---:|
| apple | 765 | 630 | 1,395 |
| banana | 749 | 632 | 1,381 |
| grape | 770 | 630 | 1,400 |
| mango | 763 | 630 | 1,393 |
| orange | 753 | 656 | 1,409 |
| **total** | **3,800** | **3,178** | **6,978** |

Manifest hash: `sha256:9e14ddad51f46a91b6526852f85251b3084f62f68e46db3dee49cb6be4fa9f39`

A mismatch means the image set changed; episode banks will refuse to load
against it and must be rebuilt.

> Note: Windows leaves `desktop.ini` files in three folders. They are excluded
> by the extension filter, so a naive `ls | wc -l` over-counts by one.

## 3. Determinism

`fsgrade/seeding.py` seeds `random`, `numpy`, `torch` (CPU + all CUDA devices),
sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`, enables
`cudnn.deterministic`, disables `cudnn.benchmark`, and requests
`torch.use_deterministic_algorithms`.

Crucially, randomness is **derived, not shared**: `derive_seed(*components)`
maps identity onto a stable seed via BLAKE2b, so an episode's randomness depends
only on its own identity — never on execution order or on how many methods ran
before it. Re-running one step cannot silently change what another step sees.

Check:

```bash
python -m fsgrade.cli.build_banks --config configs/experiment/main_loso.yaml
# rebuild and compare content_hash in results/_episodes/*.meta.json — must be identical
```

## 4. Running

```bash
bash scripts/run_all.sh --smoke                       # minutes, CPU, every code path
bash scripts/run_all.sh --stage e1                    # the pivotal control only
bash scripts/run_all.sh --data-root D:/Datasets/FruitVision   # everything
```

Individual stages:

```bash
python -m fsgrade.cli.check_data     --config configs/experiment/main_loso.yaml
python -m fsgrade.cli.build_banks    --config configs/experiment/main_loso.yaml
python -m fsgrade.cli.cache_features --encoders dinov2_vits14 clip_vitb16
python -m fsgrade.cli.evaluate       --config configs/experiment/main_loso.yaml
```

Override anything from the CLI:

```bash
python -m fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
       --set model.freeze.stage=2 train.contrastive_weight=0.0 \
       --shots 1 3 5 10 --methods ours protonet
```

Unknown override keys are an **error**, not a silent no-op — a typo cannot make
you believe you ran a sweep you did not run.

## 5. Tests

```bash
python -m pytest tests/ -q          # 192 tests, ~55 s, no GPU, no dataset needed
```

They use a synthetic dataset built in a temp directory. Notable guarantees:

- freeze ladder parameter counts are asserted **literally** (`{-1:0, 0:9_536,
  1:157_504, 2:683_072, 3:2_782_784, 4:11_176_512}`), and the historical buggy
  value (4,805,952) is asserted **not** to correspond to any valid stage;
- episode banks are deterministic, disjoint, nested and species-balanced;
- every registered method returns real logits (shift-invariant argmax, rows that
  do not sum to 1) and produces non-zero encoder gradients;
- unpaired result series **raise** rather than being truncated into a fake pairing;
- support embeddings are provably unaffected by the query batch.

## 6. Known limitations

- **Chance is 50 %.** Two-way, balanced queries. Read every accuracy against 50,
  not 0. Balanced accuracy, MCC and AUC characterise it better.
- **One dataset (being addressed).** All five FruitVision species share a capture
  setup, so the cross-species gap is narrower than a true cross-dataset shift.
  FruitNet is being added as a second dataset — see `configs/data/fruitnet.yaml`,
  which also documents the code change still needed in
  `fsgrade/data/episodes.py` before genuine train-here/test-there transfer works.
- **Fold-level n = 5.** The interval that supports the generalisation claim is
  necessarily wide with five species. It is reported anyway. FruitNet's six
  species give n = 6 on that dataset; the two intervals are **not** comparable
  and must not share a results column.
- **Dataset provenance unresolved.** The original Kaggle link has no dataset
  slug; the exact source and license must be identified before submission.
  FruitNet carries a real DOI, which limits the blast radius but does not fix it.
- **No MAML / gradient-based meta-learning baseline.** Worth acknowledging.
- **`ours` is an alias of `protonet_temp`.** Identical class, identical kwargs,
  no config distinguishes them (see the warning in `fsgrade/methods/trained.py`).
  Any delta between those two rows is seed noise. Resolve before the results
  chapter — either differentiate `ours` or drop it and make `protonet_temp` the
  reference method.
- **Experiments cut for budget.** E6b/c, E7, E9, E10 and E11 are out of scope for
  this submission. E7 (prompt-template sensitivity), E9 (robustness corruptions)
  and E10 (error analysis, Grad-CAM) have **no implementation at all** — no
  prompt-sweep CLI, no corruption code, no Grad-CAM module. They are future work
  and must not be described as run.
- **`fsgrade/analysis/` is empty.** Table and figure generation is unwritten, so
  every results artifact still has to be produced by hand or by new code.

## 7. Superseded results

Numbers produced before this refactor are **not comparable** to new ones. The
freeze depth, the baseline logits, the training harness and the episode sampler
all changed, and the previous paired statistics were computed on unpaired data.
Anything already written into the LaTeX thesis must be regenerated.
