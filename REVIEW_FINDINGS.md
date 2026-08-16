# Adversarial Review — Findings

A hostile read of the repository and thesis draft, written as an examiner would
approach it. Severity-ranked. Every finding carries evidence and a recommended
response.

Reviewed 2026-08-16 against commit `c1d4301` and the LaTeX draft at
`~/Desktop/AmrSamir_MasterThesis/` (compiled 2026-07-27).

**Status key** — `FIXED` applied in this session · `NEEDS RERUN` requires the GPU
box · `NEEDS WRITING` a text change · `OPEN` undecided.

---

## Summary

The contribution is defensible. The evidential chain supporting it is not, yet.
Nothing here is a research failure; everything here is a provenance, reporting or
consistency failure, and all of it is repairable inside the two-week window.

The single most dangerous finding is **C1**. An examiner who notices it first
will re-read everything else in bad faith.

---

## Critical — resolve before submission

### C1. Reported numbers have no reproducible producer
**Status: FIXED (mechanism) / NEEDS RERUN (numbers)**

No code in `src/` or `notebooks/main_experiment.ipynb` ever calls `json.dump`,
yet `results/*.json` exists. The producers were gitignored helpers
(`inject_outputs.py`, `extract_results.py`, `finalize_fresh.py` —
`.gitignore:46-58`) that are no longer present.

Worse, the notebook outputs were **injected rather than executed**:
`execution_count` is 12, 13 and 14 **twice each** (cells 21/23/25 and 27/29/31).
A linear execution cannot produce duplicate counts. The commit message says as
much: *"Embed computed outputs … so results render inline on GitHub."*

*Why it is critical:* a thesis's results must be traceable to a computation. As
it stands, the honest answer to "how do I regenerate Table 5.3?" is "you cannot."

*Response:* `scripts/reproduce_thesis.py` now regenerates every number, figure
and checkpoint from one invocation, writing `metrics.json`, `env.json`,
`summary.csv` and `figures/` into a timestamped run directory. Run it for seeds
42, 1337, 2024 and quote only that output. Then either re-execute the notebook
top to bottom or drop it from the deliverable in favour of the script.

### C2. 40.7% of the backbone was frozen without that being described
**Status: FIXED (mechanism) / NEEDS RERUN (numbers)**

`src/models.py` froze parameters by substring match on
`("layer1", "conv1", "bn1")`. In torchvision ResNets that also matches
`layer2.0.conv1`, `layer2.0.bn1`, `layer3.0.conv1`, … — the first convolution
and its norm in **every block of every stage**.

Measured, not inferred:

```
trainable, substring freezing = 6,765,569   <- matches the notebook exactly
trainable, stem+layer1 only   = 11,414,017
unintentionally frozen        = 4,648,448   (40.7%)
```

The 6,765,569 figure reproduces the notebook's reported parameter count to the
digit, which confirms every result came from the over-frozen model.

*Why it is critical:* the thesis describes "selectively frozen early layers to
preserve generalizable low-level features." Freezing the first conv of every
block is not that. The description and the artifact disagree, and the
`−Frozen layers` ablation (−1.7 points) was measuring something other than what
it claims.

*Response:* freezing is now explicit and selectable —
`freeze_mode="stem_layer1"` (default, matches the description) and
`freeze_mode="legacy_substring"` (reproduces the checkpoint). Re-run under the
default. **Report both as an ablation**: this converts a defect into a finding
about how much backbone plasticity cross-species transfer actually needs.

### C3. A one-sided test reports "significantly worse" as "no difference"
**Status: FIXED (mechanism) / NEEDS RERUN (numbers)**

`src/experiments.py:648` called `wilcoxon(..., alternative="greater")`, which can
only ask *"is ours better?"*. Every baseline that **beat** the proposed method
returned p = 1.00, and the `Sig?` column printed `No` — read naturally as "no
significant difference."

The paired t-tests say otherwise: Siamese t = −5.51 (p ≈ 5.4e-08), Matching
t = −4.97 (p ≈ 8.6e-07). The method is significantly **worse** than two
baselines, not tied with them. `RESULTS.md:17` nonetheless bolds
**Ours (Full Model)** as though it were the winner.

*Why it is critical:* this is the finding most likely to be read as deliberate.
It is almost certainly an honest slip — the abstract already concedes the
architecture claim — but the tables must stop contradicting the prose.

*Response:* now two-sided, with an explicit `outcome` column reading
`ours better` / `ours WORSE` / `ns` instead of an ambiguous yes/no. Un-bold
`RESULTS.md:17` and state the losses plainly.

### C4. Thesis and repository report different numbers
**Status: NEEDS RERUN + NEEDS WRITING**

| Quantity | Thesis (27 Jul) | Repo `RESULTS.md` (16 Aug) |
|---|---|---|
| Overall | 86.3% | 85.4% |
| Mango | 82.2% | 80.8% |
| Orange | 90.6% | 90.1% |
| Species CV | 79.7 ± 4.6 | 79.9 ± 4.0 |
| Ours vs ProtoNet | 84.1 vs 84.7 | — |

Two live result sets with no statement of which supersedes the other.
`REPRODUCIBILITY.md` §7 gestures at "superseded results" without resolving it.
Two paper drafts also exist (`~/Downloads/Cross_Domain_…`,
`~/Downloads/Cross_Species_…`) and may carry a third set.

*Response:* declare the Phase 1 re-run authoritative. Update `abstract.tex`,
`results.tex`, `conclusion.tex`, `introduction.tex` and both paper drafts from
the same `metrics.json`. Verify mechanically — extract every number from the
LaTeX and assert each appears in the JSON.

### C5. README denies the results that RESULTS.md asserts
**Status: NEEDS WRITING**

`README.md:220-223`: *"The thesis experiments have not been run yet, and no
result numbers are claimed here."* Committed alongside a `RESULTS.md` claiming a
complete result set. Both in commit `c1d4301`.

*Response:* one story — `src/` produced the thesis results; `fsgrade/` is an
engineering deliverable whose experiments have not been run. Say exactly that in
both files.

---

## Major — will be raised in the viva

### M1. Single seed everywhere
**Status: FIXED (mechanism) / NEEDS RERUN**

Every `±` in the thesis reflects episode or trial resampling of **one trained
model**. No claim about stability under retraining is supported. `config.py` and
the notebook pin seed 42 and nothing else.

*Response:* the driver takes `--seed`; `--aggregate` combines runs into
`mean ± sd over N seeds`. Run three seeds and report seed-level spread alongside
episode-level CIs, clearly labelled as different quantities.

### M2. The headline CI is a different quantity from every other CI
**Status: FIXED**

`test_on_unseen_fruits` computed its interval over **5 trial means**; every
baseline and ablation computed over **600 episodes**. Both printed as `±`.
Averaging 600 episodes before taking the interval removes almost all variance,
so the headline interval was the narrowest number in the thesis by construction.
`docs/evaluation_protocol.md:99-102` identifies this as the defect most needing
correction — and the current headline reproduces it anyway.

*Response:* both are now computed and returned with explicit `n`
(`ci_95_episode`, `ci_95_trial`). Quote the episode-level interval so it is
comparable with the baseline tables.

### M3. The missing fine-tuning control
**Status: FIXED (implemented) / NEEDS RERUN**

`VIVA_PREP.md` calls this "your most exposed omission" and records that there is
currently no answer. If borrowed refinements fail to beat vanilla ProtoNet, the
obvious next question is whether plain fine-tuning beats all of them. The CD-FSL
literature (Chen et al. 2019, Tian et al. 2020) repeatedly finds it does.

*Response:* `src/experiments.py:transfer_controls` now trains one conventional
ResNet-18 + linear fresh/rotten head on the seen species — ordinary mini-batches,
no episodes — and evaluates two controls from it on the same episode stream as
every other baseline, so they are paired into the significance tests:

- **Fine-tuned + Nearest Centroid** — discard the head, embed the support set,
  classify queries by nearest centroid. The Baseline++/SimpleShot control.
- **Supervised transfer (zero-shot)** — apply the trained classifier directly to
  unseen species with *no support set at all*. This is the sharper control: it
  decides whether the few-shot framing is **necessary**. If a plain binary
  classifier crosses the species boundary unaided, episodes buy nothing.

**If either wins, report that it wins.** It is consistent with the thesis's
existing honesty and with the CD-FSL literature, and it is a stronger result
than a marginal architectural claim.

### M4. Best backbone and best training set are omitted from RESULTS.md
**Status: NEEDS WRITING**

Present in the notebook, absent from `RESULTS.md`:

- EfficientNet-B0: **92.3% ± 0.4** (4.8M params) vs ResNet-18's 85.7% (6.8M) —
  smaller *and* 6.6 points better.
- Banana-only training: **90.9%** vs multi-species 85.0% — training on one
  species beats the multi-species regime the thesis advocates.

*Why it matters:* both undercut headline choices, and both are discoverable in a
committed notebook. Omitting them reads as selective reporting.

*Response:* report both, with discussion. The banana-only result is genuinely
interesting — it suggests species *diversity* in training is not what drives
transfer, which is a finding, not an embarrassment.

### M5. Split-selection sensitivity
**Status: NEEDS WRITING**

Species-CV folds range 74.3–87.2% (mean 79.9, sd 4.0). The headline split
(apple/banana/grape → mango/orange) scores 85.3% — **2nd best of 10**.
`docs/methodology.md:78-81` concedes it "sits near the favourable end", but
`RESULTS.md` presents 85.4% as *the* result.

*Response:* lead the results chapter with the CV mean as the headline and
present the fixed split as one fold. This is a stronger claim, not a weaker one:
it is the number that survives the "you picked an easy split" objection.

### M6. Baselines were trained under a worse budget — and still won
**Status: NEEDS WRITING**

`src/experiments.py:136-139`: baselines get 20 epochs, a single LR of 5e-5 across
all parameters, no freezing. "Ours" gets 30 epochs, discriminative LRs and
frozen early layers. `docs/methodology.md:268-274` admits this is "not fair to
the baselines."

*Response:* either equalise the budget in the re-run, or state prominently that
the baselines were *handicapped and still outperformed the proposed method*,
which strengthens rather than weakens the honest conclusion.

---

## Moderate

### D1. Demo cannot load the trained model
**Status: FIXED, by a different route than first planned**

`fsgrade/demo/discovery.py:318-350` enumerates arms by scanning a **two-level**
`results/<experiment>/<run>/metrics.json` tree with sibling
`checkpoints/<fold>/<method>/<hash>.pt`. No such directory existed, so every
checkpoint-backed arm was disabled, leaving only `chance` and `nc_pixel`.

Matching that layout is **not sufficient**, which I established by reading the
loader rather than assuming. `fsgrade/demo/checkpoints.py:307-330`
(`load_for_inference`) rebuilds an *fsgrade* architecture via `build_module()`
and then calls `load_state_dict(state, strict=True)`. A `src/`
`PrototypicalNetwork` has different parameter names and shapes — its projection
head is `Linear(512,1024) → Norm → ReLU → Dropout → Linear(1024,256)` under
`encoder.projection.*` — so that call fails by construction. Bridging it would
require a `src/`-specific adapter *inside* `fsgrade/`, which contradicts scoping
`fsgrade` as an engineering deliverable rather than a results source.

*Response:* the thesis model gets its own demo — `scripts/demo_app.py`, FastAPI,
fully self-contained (inline CSS/JS, images as `data:` URIs, no external
requests). It draws a **live** episode on a held-out species: samples a support
set, forms prototypes, classifies queries, and renders every support and query
image with its prediction, confidence and correctness. The driver writes
`model_bundle.pt` per run for it to load.

```bash
python scripts/demo_app.py --selfcheck        # verify without a browser
python scripts/demo_app.py                    # serve on :8000
```

`fsgrade/demo` remains as-is and still demonstrates `fsgrade` methods; it is
simply not the vehicle for the thesis model.

### D2. Same model, four different accuracies
**Status: FIXED (mechanism) / NEEDS RERUN**

85.4 (headline) / 85.6 (baseline table) / 85.7 (ablation reference) / 85.0
(single-species table) — all the same model on the same test set, differing only
by episode resampling, presented as distinct table entries.

*Response:* the driver evaluates once and reuses the result across tables.

### D3. Figure disagreed with the text beside it
**Status: FIXED**

`Imgs/confusion_matrices.png` totals 5,170/6,000 = 86.2% (fresh recall 0.809);
the `classification_report` printed by the same notebook cell says 85% / 0.79.
They came from different runs.

*Response:* `plot_confusion_matrices` now returns the matrix **and** the report
computed from the same prediction arrays, so they cannot diverge again.

### D4. Episode sampling depended on cell execution order
**Status: FIXED**

`src/dataset.py` used the global `random` module, so any episode drawn anywhere
advanced the same stream. Re-running one notebook cell shifted every subsequent
number, which is why "the same" experiment yielded 85.0–85.7.

*Response:* the dataset now owns a `random.Random(seed)` instance, with
`reseed()` for exact repeats.

### D5. Latent support→query leak in the sampling fallback
**Status: FIXED**

`src/dataset.py:98-101` padded short query sets via
`random.choices(all_images, ...)` — with replacement, from the full list
**including the support images**. Unreachable at K ≤ 10 with ~630 images/class,
but it is a leak waiting for a larger K.

*Response:* the branch now raises rather than leaking.

### D6. Transductive coupling in the episodic forward pass
**Status: FIXED — but it did not affect the reported numbers**

`src/models.py` concatenated support and query into one forward pass, so the
projection head's `BatchNorm1d` computed statistics over both.

Measured: under `model.eval()` — which is what `evaluate()` uses and therefore
what produced every reported number — swapping a query's batch-mates changes its
embedding by exactly `0.00e+00`, because BatchNorm uses running statistics. The
coupling affected **training dynamics only**.

*Response:* support and query are now embedded separately. **Do not overclaim
this as a leak fix in the thesis** — say precisely that it affected training,
not evaluation. An examiner who checks will respect the precision.

### D7. Dataset provenance — repo docs only; the thesis was already correct
**Status: RESOLVED**

`README.md:202-204` and `REPRODUCIBILITY.md:159-162` recorded the dataset as
uncited with unknown licence and "not yet citable". **The thesis does not have
this problem** — `bachelor.bib` already carried `bijoy2025fruitvision`, cited at
`methodology.tex:36` and `results.tex:71`, and methodology.tex already explains
why the formalin-treated class is excluded. The defect was in the repository
documentation, which had fallen behind the thesis.

FruitVision is a published benchmark: Bijoy et al., *Data in Brief* 61:111752
(2025), `10.1016/j.dib.2025.111752`, **CC BY-NC-ND 4.0**. Five species matching
exactly; 10,154 original images across fresh/rotten/formalin-mixed.

*Applied:* README updated with the full citation, licence and subset
description. The bib entry upgraded from `@misc` (Mendeley mirror) to `@article`
(the *Data in Brief* article of record), with the licence recorded.

*Still worth adding to the thesis (two sentences):*
1. That the **non-augmented originals** were used. The augmented release would
   put augmented copies of one source photograph in both support and query sets.
   This is a real methodological strength and currently goes unstated.
2. That **CC BY-NC-ND carries a NonCommercial term**, which sits awkwardly beside
   the Industry 4.0 deployment framing. One sentence in the limitations.

### D8. Novelty claim was unverified against 2025–26 work
**Status: RESOLVED — see `NOVELTY_ASSESSMENT.md`**

The held-out-species gap survives, but four near-neighbours must be cited, and
`introduction.tex:24`'s absolute phrasing must become a positioned claim.

---

## Minor

| # | Finding | Response |
|---|---|---|
| N1 | `config.py:15` hardcoded `~/Desktop/Amr Samir`, which does not exist | **FIXED** — `$FRUITVISION_ROOT` with repo-relative fallback |
| N2 | `RESULTS_DIR = "./results"` resolved relative to `notebooks/`, so all five figure lookups printed `[Missing]` and `Imgs/` was populated by hand | **FIXED** — absolute, repo-anchored |
| N3 | `.gitignore` excludes `results/` and `checkpoints/`; committed outputs were force-added and future runs are silently untracked | Decide deliberately; the driver writes to `results/run_*` |
| N4 | 96 MiB checkpoint committed to plain git, no LFS | Acceptable for submission; note it |
| N5 | `generate_thesis_summary` checks for `baseline_comparison.png`, which no function produces | Remove the check or write the figure |
| N6 | `Config.BATCH_SIZE = 1` unused; `configs/base.yaml:90` says 32; neither is wired to the other | Delete one |
| N7 | `fsgrade/analysis/` is an empty package | Declare as future work |
| N8 | `ours` is a literal alias of `protonet_temp` (`fsgrade/methods/registry.py:20-23`), yet `main_loso.yaml` sets it as `stats.reference_method` | Irrelevant if `fsgrade/` is scoped out; otherwise fix |
| N9 | SAP — `fsgrade`'s headline method — has never executed (`timm`/`open_clip` null) | Declare as unrun |
| N10 | E8 unwired; E7/E9/E10 unimplemented | Declare as future work |
| N11 | `simonyan2014very` is ICLR 2015; `dosovitskiy2020image` key/year mismatch | Fix in `bachelor.bib` |
| N12 | PDF metadata: empty `/Title()` `/Author()`, `CreationDate D:20090707085533` | Fix hyperref setup |
| N13 | Submission-date placeholder (Sept 2026) | Fill in |
| N14 | `results.tex:197` unresolved `TODO` from a truncated notebook log | Resolve from the re-run |
| N15 | `src/dataset.py` used an O(n²) `p not in support_paths` scan per class per episode | **FIXED** — removed with D5 |
| N16 | `.mcp.json` ships dev scaffolding in the thesis artifact | Harmless; remove if tidying |

---

## What survives scrutiny

Stated plainly, because a review that only lists defects misrepresents the work:

- **The problem formulation is a genuine gap** and holds against a 2024–2026
  sweep. See `NOVELTY_ASSESSMENT.md`.
- **The 10-partition species-split CV** is more rigorous than most published work
  in this application area, and it answers the strongest objection available
  ("you picked an easy split") with evidence rather than argument.
- **The architecture claim was withdrawn voluntarily.** The abstract reports that
  ProtoNet++ does not beat vanilla ProtoNet at matched budget. Finding the
  confound in one's own work and reporting it is the behaviour examiners want,
  and it aligns the thesis with Chen et al. and Tian et al. rather than against
  them.
- **The `fsgrade/` framework and `docs/`** are of a standard well above the
  executed pipeline — content-hashed episode banks, typed intervals that cannot
  exist without `ci_method`/`n`/`unit`, `assert_single_factor` for ablations,
  Holm correction, 192 passing tests requiring neither GPU nor dataset. Present
  it as an engineering deliverable; do not pretend it produced results.

## Recommended order of work

1. Re-run on the GPU box — `--verify-data`, then `--smoke`, then seeds 42/1337/2024.
2. Add the fine-tuning control (M3).
3. Update `RESULTS.md` and `README.md` from the run (C3, C4, C5, M4).
4. Reconcile the LaTeX (C4) and add the citations (D7, D8).
5. Verify the demo (D1).
6. Re-audit this document against the fixed state.
