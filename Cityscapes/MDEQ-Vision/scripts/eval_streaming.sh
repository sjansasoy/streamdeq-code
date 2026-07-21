#!/usr/bin/env bash
# Streaming segmentation evaluation on Cityscapes. Run from Cityscapes/MDEQ-Vision.
# Lookback is controlled by DATASET.TEST_SET (val_sequence_{N}_frame.lst); the {M}i suffix
# is the number of iterations/steps per frame. The checkpoint (TEST.MODEL_FILE) is set
# inside each config file.

# IL-StreamDEQ: 5-frame lookback, 8 solver iterations
python tools/seg_test.py --cfg experiments/cityscapes/IL-StreamDEQ/seg_mdeq_XL_5f_8i.yaml

# UR-StreamDEQ: 5-frame lookback, 2 iterations per frame (run with UR-trained weights)
# python tools/seg_test.py --cfg experiments/cityscapes/UR-StreamDEQ/seg_mdeq_XL_unroll_5f_2i.yaml

# SUR-StreamDEQ: identical streaming inference to UR-StreamDEQ; use a UR config with SUR-trained weights
# python tools/seg_test.py --cfg experiments/cityscapes/UR-StreamDEQ/seg_mdeq_XL_unroll_5f_2i.yaml

# Baseline (single frame)
# python tools/seg_test.py --cfg experiments/cityscapes/baseline/seg_mdeq_XL_sf_27i.yaml
