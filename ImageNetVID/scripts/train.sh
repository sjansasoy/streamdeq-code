#!/usr/bin/env bash
# Object detection training on ImageNet VID. Run from ImageNetVID.

# Baseline MDEQ (also used by IL-StreamDEQ at inference), single GPU
python tools/train.py configs/baseline/faster_rcnn_mdeq_fpn_1x_imagenetvid.py

# Multi-GPU (e.g. 4 GPUs)
# bash tools/dist_train.sh configs/baseline/faster_rcnn_mdeq_fpn_1x_imagenetvid.py 4

# UR-StreamDEQ (unrolled, 20 steps)
# python tools/train.py configs/UR-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20.py

# SUR-StreamDEQ (stochastically-unrolled; the _stoch config lives alongside UR)
# python tools/train.py configs/UR-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20_stoch.py
