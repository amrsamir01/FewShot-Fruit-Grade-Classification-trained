# Related work and positioning

Gathered via the arXiv and OpenAlex MCP servers configured in `.mcp.json`.
Verify every entry before it enters the bibliography — arXiv IDs are given so
they can be re-fetched.

**Audit, 2026-08-04.** All previously-flagged forward-dated arXiv IDs were
re-fetched and confirmed real: LMP 2602.18811, CLIP-SPM 2512.19036,
HyCal 2604.15678, Kanulla 2601.13665, Zhang IDR 2511.14279. Note HyCal is
few-shot *class-incremental* learning, so check its one-line summary below still
says what you mean. Two papers that invalidate earlier positioning claims were
added in §4 and §5; see the audit notes there.

---

## 1. Few-shot classification foundations

| Work | Relevance |
|---|---|
| Snell et al., *Prototypical Networks*, NeurIPS 2017 | The base method. Class prototypes as support means, Euclidean distance logits. |
| Vinyals et al., *Matching Networks*, NeurIPS 2016 | Attention over the support set. Implemented as a baseline (arXiv 1606.04080). |
| Koch et al., *Siamese Networks for One-shot Learning*, ICML-W 2015 | Pairwise relation baseline. |
| Finn et al., *MAML*, ICML 2017 | Gradient-based meta-learning. **Not implemented** — worth acknowledging as a limitation. |

## 2. The "simple baselines are hard to beat" line — the most important context

This is the literature that motivates the zero-shot control (E1) and the frozen
foundation-model arms. A 2026 thesis that omits it invites the obvious question.

| Work | Finding |
|---|---|
| Tian et al., *Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?*, ECCV 2020 (arXiv 2003.11539) | A supervised representation + linear classifier **outperforms** state-of-the-art meta-learning. |
| Chen et al., *A Closer Look at Few-shot Classification*, ICLR 2019 | Baseline++ competitive with meta-learning; the gap narrows sharply under domain shift. |
| Chen et al., *Meta-Baseline*, ICCV 2021 (arXiv 2003.04390) | Whole-classification pre-training yields embeddings comparable to meta-learning. |
| Hu et al., *Pushing the Limits of Simple Pipelines for Few-Shot Learning (PMF)*, CVPR 2022 (arXiv 2204.07305) | Pre-train → meta-train → fine-tune with a transformer backbone is state of the art on Meta-Dataset and CDFSL. **Directly motivates adding DINOv2/CLIP.** |
| Shen et al., *Partial Is Better Than All*, AAAI 2021 (arXiv 2102.03983) | Freezing/fine-tuning *particular* layers beats both extremes — motivates the freeze-depth sweep. |

## 3. Cross-domain few-shot learning

| Work | Relevance |
|---|---|
| Guo et al., *A Broader Study of Cross-Domain Few-Shot Learning* (BSCD-FSL), ECCV 2020 | The canonical CDFSL benchmark; **plain fine-tuning beats meta-learning as the domain gap widens**. Source of the `finetune_supervised` baseline. |
| Zhao et al., *Dual Adaptive Representation Alignment*, TPAMI 2023 (arXiv 2306.10511) | Prototype and distribution alignment for CDFSL. |
| Heidari et al., *Adaptive Parametric Prototype Learning*, 2023 (arXiv 2309.01342) | Learns prototypes parametrically rather than as means. |
| Zhou et al., *Meta-Exploiting Frequency Prior for CDFSL*, NeurIPS 2024 (arXiv 2411.01432) | Frequency decomposition as a transferable prior. |
| Zhang et al., *Intermediate Domain Reconstruction for CDFSL*, 2025 (arXiv 2511.14279) | Intermediate domain proxies for large gaps. |
| Rao et al., *Style Transfer-based Task Augmentation*, 2023 (arXiv 2301.07927) | Style augmentation for domain generalisation. |

**Positioning note.** CDFSL benchmarks model a *marginal* shift (different
sensor/style/modality). CSQG has a mild marginal shift but a substantial
**conditional** shift in *P(x | y)*, with the label set held fixed. This is a
distinct regime and is the framing of contribution C1.

## 4. Vision-language and text-anchored prototypes — the basis of SAP

| Work | Relevance |
|---|---|
| Radford et al., *CLIP*, ICML 2021 | The text encoder that supplies the species-conditional prior. |
| Oquab et al., *DINOv2*, TMLR 2024 | Strong frozen self-supervised features. |
| Bendou et al., *Inferring Latent Class Statistics from Text*, 2023 (arXiv 2311.14544) | Text-derived statistics predict the mean and covariance of visual class distributions; notes CLIP's cross-domain limits in few-shot settings. |
| Huang et al., *LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP*, CVPR 2024 (arXiv 2404.02285) | **Learnable class-wise blending of visual prototypes and text embeddings** — the formulation SAP follows. |
| Goswami et al., *Cross-Modal Prototype Alignment and Mixing for Training-Free Few-Shot Classification*, 2026 (arXiv 2603.24528) | **Analyses image/text prototype mixing from a bias–variance perspective and shows it "acts like a shrinkage estimator."** This is SAP's stated motivation, already published. Must be cited *before* SAP is introduced — see `methodology.md` §4. |
| Jaykumar P et al., *Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning*, 2023 (arXiv 2307.03073) | Image + text prototypes in a prototypical framework, training-free and fine-tuned variants. Direct ancestor of the SAP formulation. |
| Shakeri et al., *Few-shot Adaptation of Medical VLMs*, MICCAI 2024 (arXiv 2409.03868) | A text-informed linear probe matches convoluted prompt learning while running far faster. |
| Wang et al., *Learning Multi-Modal Prototypes for CD-FSOD*, 2026 (arXiv 2602.18811) | Dual-branch text + visual prototypes for cross-domain detection; text encodes domain-invariant semantics, visual exemplars supply domain-specific detail. **The closest conceptual precedent.** |
| Li et al., *CLIP-SPM*, 2025 (arXiv 2512.19036) | Semantic prototype modulation for visually similar classes. |
| Lee et al., *HyCal*, 2026 (arXiv 2604.15678) | Training-free prototype calibration on frozen CLIP embeddings. |
| Zhou et al., *CoOp / CoCoOp*, IJCV 2022 | Prompt learning — the alternative to prompt ensembling; a natural E7 extension. |
| Bär et al., *Frozen Feature Augmentation*, CVPR 2024 (arXiv 2403.10519) | Augmenting in frozen feature space improves few-shot transfer — cheap potential addition. |

## 5. Agricultural / food-quality vision

| Work | Relevance |
|---|---|
| **Mothkur et al., *AgroQuali-FSL: a few-shot deep learning framework with QualiProtoNet for automated quality grading of fruits and vegetables*, Scientific Reports 16(1), May 2026 (doi 10.1038/s41598-026-52715-0)** | **The nearest published work to this thesis.** CNN-Transformer encoder + prototype head, episodic *N*-way *K*-shot, decision fusion of backbone confidence with prototype distance, Grad-CAM. Evaluated on the *Fruits Fresh and Rotten* dataset — the same dataset lineage as FruitVision. Reports ~3–7 % gains over CNN/ViT/ResNet in 1- and 5-shot settings. See the differentiation table below. |
| **Jain et al., *Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction*, 2025 (arXiv 2511.01449)** | **VLMs applied to fruit freshness in both zero-shot and few-shot settings.** Proposes MAOML, a model-agnostic ordinal meta-learning algorithm exploiting label ordinality; 92.71 % accuracy averaged across fruits. Fine-grained ordinal shelf-life rather than binary grading, generative VLMs rather than dual-encoder prototypes, and no species holdout. |
| Wang & Wang, *Plant Leaves Classification: Few-Shot Learning with Siamese Networks*, IEEE Access 2019 (doi 10.1109/ACCESS.2019.2947510) | FSL in the plant domain; closest agricultural precedent. |
| Li et al., *Plant Disease Detection and Classification by Deep Learning — A Review*, IEEE Access 2021 | Survey; establishes that cross-species generalisation is under-studied. |
| Kanulla et al., *Transformer-based Multi-task Fusion for Food Spoilage Detection*, 2026 (arXiv 2601.13665) | CNN+DeiT fusion for spoilage and shelf-life. No VLM, no cross-species protocol. |
| Wei et al., *Benchmarking In-the-wild Multimodal Disease Recognition*, 2024 (arXiv 2408.03120) | Multimodal (text + visual) prototypes for plant disease, with few-shot and training-free arms. The nearest agricultural precedent for the SAP formulation. |
| Meshram & Patil, *FruitNet: Indian fruits image dataset with quality*, Data in Brief 40 (2021) 107686 (doi 10.1016/j.dib.2021.107686; Mendeley 10.17632/b6fftwbr2v) | The second dataset. 14,700+ images, 6 species (apple, banana, guava, lime, orange, pomegranate), Good/Bad/Mixed, indoor **and** outdoor capture. Public domain and properly citable. |
| Bresilla et al., *Single-Shot CNNs for Real-Time Fruit Detection*, Front. Plant Sci. 2019 | Fruit detection, not quality grading. |

### The gap this thesis fills

Vision-language methods **have** been applied to fruit freshness (Jain et al.,
2025), and few-shot prototypical grading of produce **has** been published
(Mothkur et al., 2026). Neither claim is available to this thesis, and any
sentence asserting otherwise must not survive into the manuscript.

What remains unaddressed is the *protocol*. No work found holds out an entire
species and tests on it, and none includes the zero-label control that
determines whether a support set is needed at all:

| Axis | AgroQuali-FSL | MAOML (2511.01449) | Kanulla (2601.13665) | This thesis |
|---|---|---|---|---|
| Holds out an entire **species**, tests on unseen | not evident | no | no | **yes (LOSO ×5)** |
| **Zero-label control** — does the support set help at all? | no | no | no | **yes (`zeroshot_supervised`)** |
| Text/VLM anchor for grading | no | yes (generative VLM) | no | **yes (CLIP dual-encoder prototypes)** |
| Cross-dataset, cross-capture transfer | no | no | no | **yes (FruitVision → FruitNet)** |
| Paired stats on frozen shared episode banks, typed CIs | no | no | no | **yes** |

> **Open verification task.** The AgroQuali-FSL row is inferred from abstract and
> secondary summaries — the full text was not retrievable at audit time. Obtain
> the PDF and confirm its evaluation protocol. **If it does hold out species, C1
> needs rework**, and that must be known before the related-work chapter is written.

The defensible position is therefore *protocol and benchmark*, not task novelty
and not method novelty:

> A cross-species grading protocol (**CSQG**) extended to a cross-capture
> transfer, a benchmark whose paired statistics are actually valid, and the
> control that determines whether few-shot adaptation is needed at all.

## 6. Statistical practice

| Work | Relevance |
|---|---|
| Demšar, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR 2006 | Wilcoxon + Holm correction as the standard protocol. |
| Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018 | Choosing tests and reporting them honestly. |
| Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021 | Sources of variance; motivates the fold-level interval. |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | ECE and temperature scaling. |
| Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 | The auxiliary loss. |
| Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*, ICML 2015 | The gradient reversal layer used for species-adversarial training (C3). |

---

## Open citation issue

The README's dataset link is a bare `https://www.kaggle.com/` homepage URL with
no dataset slug, so **FruitVision is not currently citable**. Before submission,
identify the exact owner/slug and license and add a proper bibliography entry —
a reviewer will ask where the data came from.

Adding FruitNet (§5) de-risks but does not close this: the thesis then has at
least one dataset with a real DOI and a stated licence, so failing to trace
FruitVision's provenance would degrade one experiment rather than invalidate the
whole empirical chapter. Still trace it.
