#!/usr/bin/env bash
set -euo pipefail

# Run two-scale partial initialization experiments for the ML project.
# Usage:
#   bash scripts/ml_project/run_phase1_two_scale.sh
# Optional:
#   GPU_ID=1 bash scripts/ml_project/run_phase1_two_scale.sh

GPU_ID="${GPU_ID:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CONFIGS=(
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale1_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale1_previous_2i.yaml"

  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale2_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale2_previous_2i.yaml"

  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale3_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale0_scale3_previous_2i.yaml"

  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale1_scale2_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale1_scale2_previous_2i.yaml"

  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale1_scale3_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale1_scale3_previous_2i.yaml"

  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale2_scale3_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_two_scale/2f_scale2_scale3_previous_2i.yaml"
)

echo "Running phase 1 two-scale experiments on GPU ${GPU_ID}"
echo "Repository directory: ${REPO_DIR}"

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running config: ${cfg}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/seg_test.py --cfg "${cfg}"
done
