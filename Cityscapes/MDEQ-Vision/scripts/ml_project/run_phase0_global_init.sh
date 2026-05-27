#!/usr/bin/env bash
set -euo pipefail

# Run global initialization experiments for the ML project.
# Usage:
#   bash scripts/ml_project/run_phase0_global_init.sh
# Optional:
#   GPU_ID=1 bash scripts/ml_project/run_phase0_global_init.sh

GPU_ID="${GPU_ID:-0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

CONFIGS=(
  "experiments/cityscapes/ml_project/phase0_global_init/2f_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_previous_4i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_previous_8i.yaml"

  "experiments/cityscapes/ml_project/phase0_global_init/2f_zero_1i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_zero_2i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_zero_4i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_zero_8i.yaml"

  "experiments/cityscapes/ml_project/phase0_global_init/2f_stale_k2_1i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_stale_k2_2i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_stale_k2_4i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/2f_stale_k2_8i.yaml"

  "experiments/cityscapes/ml_project/phase0_global_init/5f_previous_1i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_previous_2i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_previous_4i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_previous_8i.yaml"

  "experiments/cityscapes/ml_project/phase0_global_init/5f_stale_k2_1i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_stale_k2_2i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_stale_k2_4i.yaml"
  "experiments/cityscapes/ml_project/phase0_global_init/5f_stale_k2_8i.yaml"
)

echo "Running phase 0 global initialization experiments on GPU ${GPU_ID}"
echo "Repository directory: ${REPO_DIR}"

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running config: ${cfg}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/seg_test.py --cfg "${cfg}"
done
