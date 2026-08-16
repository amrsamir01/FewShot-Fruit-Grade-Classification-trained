# Methodology

Draft source for the thesis methodology chapter. Every claim here is either
implemented in `fsgrade/` or flagged as pending measurement.

---

## 1. Task definition: Cross-Species Quality Grading (CSQG)

Let *S* be a set of **seen** fruit species with full fresh/rotten supervision, and
*u* ∉ *S* an **unseen** species. Given *K* labelled examples per quality class of
species *u*, predict the quality of unlabelled images of *u*.

Formally, for quality label *y* ∈ {fresh, rotten} and species *s*:

| Quantity | Behaviour across species | Consequence |
|---|---|---|
| label set 𝒴 | **invariant** | classes are shared, not novel |
| label semantics | **invariant** | "rotten" means the same thing for a mango as for a banana |
| class-conditional appearance *P(x \| y, s)* | **species-dependent** | rotten banana = brown-black; rotten orange = white-green mould; rotten grape = shrivelled |
| class prior *P(y \| s)* | approximately constant | dataset is near-balanced per species |

### Why this is not standard few-shot classification

Standard FSL holds out *classes*: the novelty is a label the model has never
seen, so a support set is logically necessary to define the new class. Here the
label set is fixed and only the species changes. Nothing about the task
*requires* a support set — a plain binary classifier trained on *S* can be
applied to *u* directly.

This has a sharp methodological consequence, and it is the single most
important experiment in the thesis:

> **E1.** Train a binary fresh/rotten CNN on the seen species and evaluate it on
> the unseen species with **no support set at all** (`zeroshot_supervised`).
> Every episodic method must be interpreted relative to this number.

The pre-refactor results hint at the answer: accuracy moved from **84.3 %
(1-shot) to 86.6 % (10-shot)** — a **+2.3 point** return on ten times the
support data. If the support set contributes so little, the prototypical
machinery must justify its existence, and the thesis should say so plainly
either way.

### Why this is not standard cross-domain few-shot learning either

CDFSL benchmarks (Guo et al., 2020) model a *marginal* shift: the target images
come from a different sensor, style or modality. Here the marginal shift is
mild — all species come from one dataset with a common capture setup — but the
**conditional** *P(x | y)* changes substantially, because what "rotten" looks
like is species-specific. Naming and characterising this conditional-shift
regime is contribution **C1**.

---

## 2. Protocol

### Primary: leave-one-species-out (LOSO)

Five folds; each of {apple, banana, grape, mango, orange} is the unseen species
exactly once. Train on the other four, holding out 15 % of their images as a
validation split for model selection.

Model selection **never** touches the held-out species. That is methodologically
correct, and it means validation accuracy on seen species is *not* predictive of
unseen-species accuracy — a gap worth reporting explicitly.

The headline statistic is the **mean over folds with a t-interval over folds
(n = 5)**. This is the interval that actually supports the claim "generalises to
an unseen species", and it is much wider than an interval over episodes.

### Secondary: C(5,3) split cross-validation

All 10 three-train / two-test partitions, retained as a robustness analysis.
**Not comparable to LOSO** — fewer training species and a different test
composition — so the two must never share a results column.

### Case study: the original fixed split

Apple, Banana, Grape → Mango, Orange, kept for continuity with the
pre-refactor figures. Note this split sits near the favourable end of the
observed range (the earlier C(5,3) sweep spanned 73.8 %–89.1 %, mean 79.7 %).

### Episodes

| Parameter | Value |
|---|---|
| N-way | 2 (fresh, rotten) — chance = **50 %** |
| K-shot | 5 by default; 1/3/5/10 evaluated from one bank |
| Query | 15 per class (balanced) |
| Episodes | 500 train / 200 val / 600 test per fold |

Episodes are **frozen artifacts**, not sampled at run time
(`fsgrade/data/episodes.py`):

1. **Shared across methods.** Every method sees byte-identical episodes, which
   is what makes the paired statistics valid.
2. **Nested supports.** The bank stores 10 shots; `with_shots(k)` takes a
   per-class prefix. The K-shot curve is therefore *paired across K* with
   identical query sets, removing sampler noise from the trend.
3. **Balanced per species.** Each species contributes exactly `n/|species|`
   episodes, so per-species intervals have a fixed, known *n*.
4. **Disjoint support/query**, enforced at build time; an under-sized pool
   raises `InsufficientImagesError` on the CPU in seconds.

---

## 3. Model

### Encoder

`fsgrade/models/backbones.py`. ResNet-18/34/50 and EfficientNet-B0 fine-tuned;
DINOv2 ViT-S/14 and CLIP ViT-B/16 used frozen with a feature cache.

**Freeze depth is an explicit, swept hyperparameter** (`fsgrade/models/freezing.py`):

| stage | frozen modules | frozen params |
|---:|---|---:|
| −1 | none | 0 |
| 0 | conv1, bn1 | 9,536 |
| 1 | + layer1 | 157,504 |
| 2 | + layer2 | 683,072 |
| 3 | + layer3 | 2,782,784 |
| 4 | + layer4 (entire backbone) | 11,176,512 |

Matching is by module identity, never substring. Frozen BatchNorm modules are
additionally forced into eval mode each epoch, because `requires_grad = False`
does **not** stop running-statistic updates.

### Prototypical head

Prototypes are class means of support embeddings; logits are negative Euclidean
(or scaled cosine) distance to each prototype, optionally divided by a learnable
temperature clamped to [0.1, 2.0].

**Support and query are encoded in separate forward passes.** A single
concatenated pass with BatchNorm in the projection head lets query batch
statistics influence the support embeddings during training — transductive
leakage, verified at Δ = 0.27 in embedding space and reduced to exactly 0 by
separate passes.

### Losses

- Cross-entropy with label smoothing (0.1). Note label smoothing systematically
  induces underconfidence, so it is a factor in the calibration discussion.
- Supervised contrastive auxiliary term, weight λ = 0.1 (Khosla et al., 2020).
- **Species-adversarial term (ablation, not a contribution):** a species
  classifier behind a gradient reversal layer (Ganin & Lempitsky, 2015), λ ramped
  on the DANN schedule. Rationale: the training set carries species labels that
  the original model discarded entirely, yet species is exactly the nuisance
  variable under CSQG. `adversarial_weight` defaults to `0.0` and this arm has
  never been run; it is a footnote until it is.

---

## 4. Proposed method: Species-Anchored Prototypes (SAP)

### Motivation

The visual prototype cannot transfer the species-conditional appearance prior,
because it is estimated from *K* target images and *K* is small. But language
can supply that prior at zero labelling cost: CLIP's text encoder produces
embeddings for `"a photo of a rotten mango"` without a single labelled mango.

### Formulation

For quality class *c* and unseen species *u*:

```
p_c^text   = normalize( mean over prompt ensemble of  E_text("a photo of a {c} {u}") )
p_c^visual = normalize( mean of support embeddings of class c )
p_c        = normalize( α_K · p_c^visual  +  (1 − α_K) · p_c^text )
logits     = scale · (E_image(q) · p_cᵀ)
```

### Shot-adaptive blending

Mixing visual and text prototypes is a **shrinkage estimator**: the text anchor
is a low-variance, high-bias estimate of the class centroid, the *K*-shot visual
mean is high-variance and low-bias, and the blend trades one against the other.
This bias–variance reading is not ours — it is established by Goswami et al.
(2026, arXiv 2603.24528), and LP++ (Huang et al., 2024) already learns class-wise
blending multipliers. We adopt it and instantiate it.

Because a *K*-shot mean has variance ~1/*K*, the weight on the visual prototype
should grow with *K*. We use the shrinkage form

```
α_K = K / (K + κ)
```

with κ calibrated by grid search **on validation episodes drawn from the seen
species only** — never on the held-out species, which would leak target
information and invalidate the cross-species claim. At *K* = 0, α = 0 and the
method degenerates exactly to zero-shot CLIP, so one model spans the whole shot
range.

### Falsifiable claim

> Text anchors supply the species-conditional prior that visual prototypes
> cannot transfer, so the advantage of SAP over a purely visual prototype is
> **largest at low K and decreases monotonically as K grows**.

Tested by **E5**: the K ∈ {0, 1, 3, 5, 10} curve for `sap` versus `clip_ncc`.

### Positioning

**SAP is an applied instantiation of an established line, not a new idea, and the
thesis must say so first.** Text-anchored prototypes are well covered — Proto-CLIP
(arXiv 2307.03073); Bendou et al. (2023); LP++ (Huang et al., 2024); LMP
(arXiv 2602.18811); CLIP-SPM (arXiv 2512.19036); HyCal (arXiv 2604.15678) — and
Goswami et al. (2026, arXiv 2603.24528) publish the exact bias–variance/shrinkage
analysis that motivates the blend. Nor is the application domain untouched:
Jain et al. (2025, arXiv 2511.01449) apply VLMs to fruit freshness in zero- and
few-shot settings, and Mothkur et al. (2026) publish episodic prototypical
grading of produce on the same dataset lineage as FruitVision.

What is defensibly ours in SAP is narrower and should be claimed precisely:

1. **κ is calibrated on seen-species validation episodes only.** No target-species
   labels are consumed by hyperparameter selection, so the cross-species claim
   survives the calibration step. Methods that tune the blend on target data
   cannot say this.
2. **K = 0 degenerates exactly to zero-shot CLIP**, so a single model spans the
   whole shot range and the K-curve is continuous through the zero-label point —
   which is what makes it directly comparable to the E1 control.

Neither is a large delta. Claim them plainly, cite the line above ahead of the
method, and let the *protocol* carry the contribution.

### De-risking

**C1 (protocol), C2 (the control) and C3 (statistical validity) stand regardless
of whether SAP wins.** If text anchoring loses to a plain fine-tuned baseline,
"few-shot adaptation contributes little over direct transfer when label semantics
are invariant" is itself a legitimate finding. The thesis must not depend on SAP
winning — and after the 2026-08-04 literature audit it must not depend on SAP
being novel either, since it is not.

---

## 5. Baseline ladder

Ordered so that adjacent rows isolate one factor
(`fsgrade/methods/registry.py`):

| Method | Target labels used | What it isolates |
|---|---|---|
| `chance` | — | the 50 % floor |
| `nc_pixel` | K | training-free pixel baseline |
| `zeroshot_supervised` | **0** | **does the task need a support set?** |
| `ncc_supervised` | K | value of the support set, same encoder |
| `finetune_supervised` | K | Chen 2019 / Guo 2020 transfer baseline |
| `siamese`, `matching` | K | episodic metric baselines (corrected) |
| `protonet`, `protonet_temp`, `ours` | K | prototypical family |
| `clip_text_zeroshot` | **0** | language only, no target images |
| `dinov2_ncc/probe`, `clip_ncc/probe` | K | frozen foundation features |
| `sap` | K (+ text) | proposed |

`zeroshot_supervised`, `ncc_supervised` and `finetune_supervised` **share one
checkpoint per fold**, so the comparison is exact: identical weights, identical
features, differing only in how the target support is used.

All trained methods go through one `Trainer` with one optimizer construction,
one schedule and one early-stopping rule. Ablation arms are checked by
`assert_single_factor`, which refuses to run an arm that differs from its base
config in more than the studied factor.

### On fairness of hyperparameters

Pinning identical learning rates across architecturally different methods is not
fair to the baselines. The intended protocol is an identical **tuning budget**
per method (same grid, selected on seen-species validation episodes), with the
grid and selected values reported in an appendix. That is defensible under
examination in a way that "we used our learning rate for everyone" is not.
