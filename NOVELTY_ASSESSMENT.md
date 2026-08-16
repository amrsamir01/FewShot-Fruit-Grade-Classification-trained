# Novelty Assessment and Literature Position

Evidence-backed verdict on the thesis's contribution claim. Closes the
`VIVA_PREP.md` item *"Phase B literature sweep — needs session restart for MCP."*

Sweep performed 2026-08-16 over OpenAlex, Crossref, arXiv and Semantic Scholar,
covering 2017–2026, plus targeted web search. Queries spanned few-shot ×
fruit/produce/postharvest, cross-species/cross-cultivar × quality/freshness/
grading, and vision-language × agriculture × few/zero-shot.

---

## 1. Verdict

**The problem-formulation claim survives. The absolute phrasing does not.**

No located work evaluates fruit *quality* classification on **held-out species** —
training on one set of botanical species and testing on species never seen during
training. That is the thesis's contribution and it stands.

But `introduction.tex:24` currently reads *"to the best of our knowledge, no
existing method…"*. Four works published since the draft was written sit close
enough that an unqualified claim invites a viva ambush. Replace it with a
positioned claim that names them (§2), and the contribution becomes stronger,
not weaker — it shows command of the field rather than absence of a search.

Confidence: **high** for the held-out-species gap; **moderate** for "nobody has
done cross-cultivar quality transfer", since cultivar-level work in horticulture
journals is poorly indexed and may exist outside the searched corpora. Phrase
accordingly.

## 2. Near neighbours that must be cited and distinguished

| Work | Venue / DOI | Why it is close | Why it does not close the gap |
|---|---|---|---|
| *Training-free few-shot strawberry appearance quality recognition via structural re-parameterization and adaptive subspace learning* | J. Food Meas. Charact. 2026 · `10.1007/s11694-026-04617-y` | **The closest hit.** Few-shot × fruit **quality** — the same two axes as this thesis | Single species throughout. No species boundary is ever crossed, so it cannot speak to transfer |
| *Few-Shot Learning for Strawberry Variety Classification: A Prototypical Network-Based Approach* | Firat Univ. J. ECE 2026 · `10.62520/fujece.1925239` | ProtoNet on fruit imagery — same method family | Classifies **variety identity**, not quality. Species/variety is the label, not the held-out axis |
| *Zero-Shot Learning Framework for 3D Texture–Geometry Defect Estimation in Mixed Fruits and Vegetables Using Signed Distance Fields* | Indian J. Sci. Tech. 2026 · `10.17485/ijst/v19i22.592` | Cross-fruit defect estimation without target labels | 3D signed-distance modality, not RGB. Localised geometric defects, not holistic appearance quality |
| *A General Machine Learning Model for Assessing Fruit Quality Using Deep Image Features* | AI 2023 · `10.3390/ai4040041` | Explicitly pursues a *general* fruit-quality model | Pools all species into one training set. No held-out-species evaluation, so generality is asserted rather than measured |

**How to phrase the contribution.** The distinguishing move is the *evaluation
protocol*, not the architecture: these four ask "can a model judge quality with
few labels?"; this thesis asks "does a quality judgment survive a species
boundary?" — and answers it with a species-disjoint split plus a 10-partition
species-split cross-validation. Lead with the protocol.

## 3. Missing canon — add before submission

The cross-domain few-shot (CD-FSL) literature is the frame an examiner will
place this work in, and `bachelor.bib` currently omits its standard references.
Their absence is conspicuous because the thesis's own negative result (a tuned
ProtoNet variant failing to beat vanilla ProtoNet) is *exactly* what this
literature predicts.

- **Guo et al., "A Broader Study of Cross-Domain Few-Shot Learning"** (ECCV 2020)
  — the BSCD-FSL benchmark; establishes that domain gap dominates method choice.
- **Chen et al., "A Closer Look at Few-shot Classification"** (ICLR 2019)
  — Baseline++; shows simple fine-tuning rivals meta-learning under domain shift.
- **Tian et al., "Rethinking Few-Shot Image Classification"** (ECCV 2020)
  — a good embedding beats sophisticated meta-learning.
- **Wang et al., "SimpleShot"** (arXiv 2019)
  — nearest-centroid on good features is a strong baseline.

These four turn the negative result from an apparent failure into a
**confirmation of an established finding in a new domain**. That is the single
highest-leverage citation change available.

## 4. Dataset provenance — RESOLVED (repo docs only)

`README.md:202-204` and `REPRODUCIBILITY.md:159-162` recorded the dataset as
uncited with unknown licence. **The thesis was already correct**: `bachelor.bib`
carried `bijoy2025fruitvision`, cited at `methodology.tex:36` and
`results.tex:71`, with the formalin-class exclusion already explained. Only the
repository documentation had fallen behind. It is a published, citable benchmark:

> **FruitVision: A benchmark dataset for fresh, rotten, and formalin-mixed fruit
> detection.** Md. Hasan Imam Bijoy, Syeda Zarin Tasnim, Syed Ali Awsaf,
> Md. Zahid Hasan. *Data in Brief*, vol. 61, art. 111752, August 2025.
> DOI: `10.1016/j.dib.2025.111752`
> Mirror: `https://data.mendeley.com/datasets/xkbjx8959c/2`

- **Species match exactly**: apple, banana, grape, mango, orange.
- **Original collection**: 10,154 images across 15 categories (5 species × 3
  classes: fresh / rotten / formalin-mixed). An augmented expansion to 81,000+
  images also ships.
- **This thesis uses 6,978 images** — the fresh/rotten subset of the *original*,
  non-augmented collection, discarding the formalin-mixed class.

Three things follow, all of which belong in the thesis:

1. **State that the non-augmented originals were used.** Training on the
   augmented release would place augmented copies of the same source photograph
   in both support and query sets — a leak. Choosing the originals is a
   methodological strength and currently goes unmentioned.
2. **Licence is CC BY-NC-ND 4.0.** Attribution is required. The NonCommercial
   term sits awkwardly beside the thesis's Industry 4.0 deployment framing —
   add one sentence to the limitations noting that commercial deployment would
   require separate licensing of the data.
3. **Justify dropping formalin-mixed.** A third class exists and was discarded.
   The binary fresh/rotten task is defensible, but say so deliberately —
   otherwise it reads as an unexamined convenience.

## 5. Framing the headline number

Within-species fresh/rotten classification is a solved problem: the literature
routinely reports **94–98%** (`10.3390/s22218192`, `10.3390/app13148087`,
`10.1007/s44163-025-00750-7`). A reader who meets ~85% cold will read it as
underperformance.

Frame it against two reference points, both of which belong in the results
chapter:

- **Floor**: 50% chance on a balanced binary task; 69.6% for non-learned
  nearest-centroid on raw pixels.
- **Ceiling**: the 94–98% within-species figures — which are *not comparable*,
  because they train and test on the same species. Say this explicitly.

The correct claim is about the *retained fraction* of within-species performance
under a species shift, not the absolute number.

## 6. Bib entries to add

```bibtex
@article{bijoy2025fruitvision,
  title   = {{FruitVision}: A benchmark dataset for fresh, rotten, and
             formalin-mixed fruit detection},
  author  = {Bijoy, Md. Hasan Imam and Tasnim, Syeda Zarin and
             Awsaf, Syed Ali and Hasan, Md. Zahid},
  journal = {Data in Brief},
  volume  = {61},
  pages   = {111752},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.111752},
  note    = {Licensed CC BY-NC-ND 4.0}
}

@inproceedings{guo2020broader,
  title     = {A Broader Study of Cross-Domain Few-Shot Learning},
  author    = {Guo, Yunhui and Codella, Noel C. and Karlinsky, Leonid and
               Codella, James V. and Smith, John R. and Saenko, Kate and
               Rosing, Tajana and Feris, Rogerio},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@inproceedings{chen2019closerlook,
  title     = {A Closer Look at Few-shot Classification},
  author    = {Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and
               Wang, Yu-Chiang Frank and Huang, Jia-Bin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}

@inproceedings{tian2020rethinking,
  title     = {Rethinking Few-Shot Image Classification: A Good Embedding Is
               All You Need?},
  author    = {Tian, Yonglong and Wang, Yue and Krishnan, Dilip and
               Tenenbaum, Joshua B. and Isola, Phillip},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@article{wang2019simpleshot,
  title   = {{SimpleShot}: Revisiting Nearest-Neighbor Classification for
             Few-Shot Learning},
  author  = {Wang, Yan and Chao, Wei-Lun and Weinberger, Kilian Q. and
             van der Maaten, Laurens},
  journal = {arXiv preprint arXiv:1911.04623},
  year    = {2019}
}
```

Also fix, per `VIVA_PREP.md`: `simonyan2014very` is ICLR 2015, not arXiv; and
`dosovitskiy2020image` has a key/year mismatch.

## 7. What did NOT turn up

Worth stating in the viva, because absence of evidence was searched for:

- No few-shot fruit-quality work using a species-disjoint protocol.
- No cross-cultivar quality-grading benchmark.
- No prior use of FruitVision for few-shot or cross-species evaluation — the
  dataset is 2025 and its uses so far are conventional supervised detection.

The last point is a genuine secondary contribution: **first few-shot benchmark
protocol defined on FruitVision.** It is modest, it is real, and it is worth one
sentence in the contributions list.
