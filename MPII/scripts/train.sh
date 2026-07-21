#!/usr/bin/env bash
# Pose estimation training on MPII. Run from MPII.

# Baseline MDEQ (also used by IL-StreamDEQ at inference), single GPU
python tools/train.py configs/baseline/mdeq_mpii_256x256_lr_1e4.py

# Multi-GPU (e.g. 4 GPUs)
# bash tools/dist_train.sh configs/baseline/mdeq_mpii_256x256_lr_1e4.py 4

# UR-StreamDEQ (unrolled, 20 steps)
# python tools/train.py configs/UR-StreamDEQ/mdeq_mpii_256x256_lr_1e4_unroll_20.py

# SUR-StreamDEQ (stochastically-unrolled; the _stoch config lives alongside UR)
# python tools/train.py configs/UR-StreamDEQ/mdeq_mpii_256x256_lr_1e4_unroll_20_stoch.py
