#!/usr/bin/env bash
# =====================================================================
#  Full thesis run, in dependency order. Resumable and idempotent.
#
#    bash scripts/run_all.sh --data-root D:/Datasets/FruitVision
#    bash scripts/run_all.sh --smoke              # minutes, CPU, all code paths
#    bash scripts/run_all.sh --stage e1           # only the pivotal control
#
#  Every stage writes to results/<experiment>/<timestamp>/ and can be re-run
#  independently. Nothing depends on a Jupyter kernel staying alive.
# =====================================================================
set -euo pipefail

DATA_ROOT="${FRUITVISION_DATA_ROOT:-D:/Datasets/FruitVision}"
STAGE="all"
SMOKE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --smoke)     SMOKE=1; shift ;;
    --stage)     STAGE="$2"; shift 2 ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

PY="python -m"
COMMON=(--data-root "$DATA_ROOT")
[[ ${#EXTRA[@]} -gt 0 ]] && COMMON+=("${EXTRA[@]}")

banner() { echo; echo "=================================================================="; echo "  $1"; echo "=================================================================="; }

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# ---------------------------------------------------------------- smoke ---- #
if [[ $SMOKE -eq 1 ]]; then
  banner "SMOKE: end-to-end pipeline on tiny episode counts"
  $PY fsgrade.cli.check_data  --config configs/experiment/smoke.yaml "${COMMON[@]}"
  $PY fsgrade.cli.build_banks --config configs/experiment/smoke.yaml "${COMMON[@]}"
  $PY fsgrade.cli.evaluate    --config configs/experiment/smoke.yaml "${COMMON[@]}"
  banner "SMOKE PASSED"
  exit 0
fi

# ------------------------------------------------------------- step 0-2 ---- #
if run_stage prep || run_stage all; then
  banner "STEP 0  Validate dataset (fails fast, before any GPU time)"
  $PY fsgrade.cli.check_data --config configs/experiment/main_loso.yaml \
      --duplicates --write-manifest "${COMMON[@]}"

  banner "STEP 1  Build deterministic episode banks (LOSO + C(5,3))"
  $PY fsgrade.cli.build_banks --config configs/experiment/main_loso.yaml "${COMMON[@]}"
  $PY fsgrade.cli.build_banks --config configs/experiment/cv_c53.yaml    "${COMMON[@]}"
fi

# ------------------------------------------------------------------ E1 ----- #
# The pivotal control. Run this FIRST: its result frames the whole thesis.
if run_stage e1 || run_stage all; then
  banner "E1  Zero-shot supervised transfer  <-- does few-shot buy anything?"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
      --methods chance zeroshot_supervised ncc_supervised ours \
      --set experiment=e1_zeroshot_control stats.reference_method=zeroshot_supervised \
      "${COMMON[@]}"
fi

# ------------------------------------------------------------- E2/E3 ------- #
if run_stage main || run_stage all; then
  banner "E2+E3  Full ladder under leave-one-species-out CV (headline protocol)"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml "${COMMON[@]}"
fi

# ---------------------------------------------------------------- E5 ------- #
if run_stage shots || run_stage all; then
  banner "E5  K-shot curve (nested supports -> paired across K, no retraining)"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
      --shots 1 3 5 10 --set experiment=e5_shots "${COMMON[@]}"
fi

# ---------------------------------------------------------------- E6 ------- #
if run_stage ablation || run_stage all; then
  banner "E6a  Freeze-depth sweep (the ablation the substring bug invalidated)"
  for stage in -1 0 1 2 3 4; do
    $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
        --methods ours --set model.freeze.stage=$stage \
        "experiment=e6_freeze_stage${stage}" "${COMMON[@]}"
  done

  banner "E6b  Component ablation"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml --methods ours \
      --set train.contrastive_weight=0.0 experiment=e6_no_contrastive "${COMMON[@]}"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml --methods ours \
      --set model.temperature.enabled=false experiment=e6_no_temperature "${COMMON[@]}"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml --methods ours \
      --set model.dropout=0.0 experiment=e6_no_dropout "${COMMON[@]}"

  banner "E6c  Species-adversarial regularisation (contribution C3)"
  $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
      --methods zeroshot_supervised ncc_supervised \
      --set train.adversarial_weight=0.3 experiment=e6_species_adversarial "${COMMON[@]}"
fi

# ---------------------------------------------------------------- E4 ------- #
if run_stage backbone || run_stage all; then
  banner "E4  Backbone comparison (completes the truncated Experiment 5)"
  for bb in resnet18 resnet50 efficientnet_b0; do
    $PY fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml \
        --methods ours --set "model.backbone=$bb" "experiment=e4_backbone_$bb" \
        "${COMMON[@]}"
  done
fi

# ------------------------------------------------------- E4b / SAP --------- #
if run_stage vlm || run_stage all; then
  banner "E4b  Frozen foundation features + SAP  (needs timm + open_clip_torch)"
  $PY fsgrade.cli.cache_features --config configs/experiment/vlm_loso.yaml \
      --encoders dinov2_vits14 clip_vitb16 "${COMMON[@]}" || \
      echo "  [skipped] install with: pip install timm open_clip_torch"
  $PY fsgrade.cli.evaluate --config configs/experiment/vlm_loso.yaml "${COMMON[@]}" || \
      echo "  [skipped] foundation models unavailable"

  banner "E5b  SAP shot curve incl. K=0 (tests the central claim)"
  $PY fsgrade.cli.evaluate --config configs/experiment/vlm_loso.yaml \
      --methods clip_text_zeroshot sap clip_ncc --shots 1 3 5 10 \
      --set experiment=e5b_sap_shots "${COMMON[@]}" || true
fi

# --------------------------------------------------------------- E11 ------- #
if run_stage cv || run_stage all; then
  banner "E11  C(5,3) split CV (secondary robustness -- NOT comparable to LOSO)"
  $PY fsgrade.cli.evaluate --config configs/experiment/cv_c53.yaml "${COMMON[@]}"

  banner "Original fixed split, retained as a case study"
  $PY fsgrade.cli.evaluate --config configs/experiment/fixed_split.yaml "${COMMON[@]}"
fi

banner "DONE -- results under results/ ; every number is recomputable from per_query.csv.gz"
