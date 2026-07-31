#!/usr/bin/env bash
# Launches the real StreamDEQ ImageNet-VID training (7 epochs, combined
# VID+DET dataset, matching the paper's setup) using this repo's own
# tools/dist_train.sh, in the environment set up by setup_train_env.sh.
#
# Usage:
#   bash scripts/launch_training.sh [num_gpus] [env_name] [extra tools/train.py args...]
#
# Examples:
#   bash scripts/launch_training.sh 4
#   bash scripts/launch_training.sh 4 streamdeq_vid_train --resume-from work_dirs/faster_rcnn_mdeq_fpn_1x_imagenetvid/latest.pth
#
# Before running, make sure the dataset is in place:
#   data/ILSVRC/annotations/{imagenet_vid_train,imagenet_det_30plus1cls}.json
#   data/ILSVRC/Data/{VID,DET}/
# and the ImageNet-pretrained MDEQ backbone at:
#   pretrained_models/mdeq_XL_cls_new.pkl
#
# Training is significantly slower than typical explicit-backbone detectors:
# each image requires solving a fixed point (F_THRES Broyden iterations) in
# both the forward and backward pass. A full run (7 epochs over the combined
# VID+DET dataset, ~1.47M images/epoch) can take on the order of weeks to
# months depending on GPU count and model -- this is expected for DEQ-style
# training, not a sign anything is broken. Checkpoints save every epoch
# (checkpoint_config.interval=1 in the config), so the run is safely
# resumable with --resume-from if interrupted.

set -e

NUM_GPUS="${1:?Usage: launch_training.sh <num_gpus> [env_name] [extra args...]}"
ENV_NAME="${2:-streamdeq_vid_train}"
shift 2 2>/dev/null || shift 1
EXTRA_ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"

echo "Launching training: $NUM_GPUS GPU(s), env '$ENV_NAME'"
echo "Config: configs/streamdeq/faster_rcnn_mdeq_fpn_1x_imagenetvid.py"
echo "(This is the non-streaming MDEQ_FasterRCNN config, matching the paper's"
echo " training setup -- streaming behavior only applies at test time.)"

bash tools/dist_train.sh \
    configs/streamdeq/faster_rcnn_mdeq_fpn_1x_imagenetvid.py \
    "$NUM_GPUS" \
    $EXTRA_ARGS
