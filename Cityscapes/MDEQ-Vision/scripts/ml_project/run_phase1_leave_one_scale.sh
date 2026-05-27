#!/usr/bin/env bash
set -euo pipefail

# Run leave-one-scale-out experiments for the ML project.
# Usage:
#   bash scripts/ml_project/run_phase1_leave_one_scale.sh
# Optional:
#   GPU_ID=1 bash scripts/ml_project/run_phase1_leave_one_scale.sh

GPU_ID="${GPU_ID:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CONFIGS=(
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale0_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale0_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale0_previous_4i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale0_previous_8i.yaml"

  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale1_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale1_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale1_previous_4i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale1_previous_8i.yaml"

  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale2_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale2_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale2_previous_4i.yaml"

  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale3_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale3_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale3_previous_4i.yaml"

  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale0_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale0_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale0_previous_4i.yaml"

  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale1_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale1_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase1_leave_one_scale/5f_drop_scale1_previous_4i.yaml"
)

echo "Running phase 1 leave-one-scale-out experiments on GPU ${GPU_ID}"
echo "Repository directory: ${REPO_DIR}"

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running config: ${cfg}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/seg_test.py --cfg "${cfg}"
done
