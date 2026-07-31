#!/usr/bin/env bash
# Segmentation training on Cityscapes. Run from Cityscapes/MDEQ-Vision.

# Baseline MDEQ (also used by IL-StreamDEQ at inference), single GPU
python tools/seg_train.py --cfg experiments/cityscapes/baseline/seg_mdeq_XL_sf_27i.yaml

# Multi-GPU (e.g. 4 GPUs)
# python -m torch.distributed.launch --nproc_per_node=4 tools/seg_train.py \
#     --cfg experiments/cityscapes/baseline/seg_mdeq_XL_sf_27i.yaml

# UR-StreamDEQ (unrolled, 20 steps)
# python tools/seg_train.py --cfg experiments/cityscapes/UR-StreamDEQ/seg_mdeq_XL_unroll_20.yaml

# SUR-StreamDEQ (stochastically-unrolled)
# python tools/seg_train.py --cfg experiments/cityscapes/UR-StreamDEQ/seg_mdeq_XL_unroll_20_stoch.yaml
