#!/usr/bin/env bash
# Streaming detection evaluation on ImageNet VID. Run from ImageNetVID.
# In the config name, {N}f = temporal lookback (reference frames) and {M}i = iterations/steps per frame.
# Point CKPT at your trained checkpoint before running.
# CKPT=work_dirs/faster_rcnn_mdeq_fpn_1x_imagenetvid/latest.pth

# IL-StreamDEQ: 5-frame lookback, 2 solver iterations
# python tools/test.py \
#     configs/IL-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_5f_2i.py \
#     ${CKPT} --eval bbox

# UR-StreamDEQ: 5-frame lookback, 2 unroll steps
# python tools/test.py \
#     configs/UR-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20_stream_5f_2i.py \
#     ${CKPT} --eval bbox

# SUR-StreamDEQ: identical streaming inference to UR-StreamDEQ; use the same config with SUR-trained weights
# python tools/test.py \
#     configs/UR-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_unroll_20_stream_5f_2i.py \
#     ${CKPT} --eval bbox
