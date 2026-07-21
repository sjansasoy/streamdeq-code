#!/usr/bin/env bash
# Streaming pose evaluation on MPII. Run from MPII.
# In the config name, {N}f = temporal lookback (num_prev_frames); {N}u = unroll steps
# and {N}i = equilibrium-solver iterations per frame.
# Point CKPT at your trained checkpoint before running.
# CKPT=work_dirs/mdeq_mpii_256x256_lr_1e4/best_PCKh_epoch.pth

# IL-StreamDEQ: 1 solver iteration, 5-frame lookback
# python tools/test.py configs/IL-StreamDEQ/mdeq_mpii_vid_256x256_1i_5f.py ${CKPT} --eval PCKh

# UR-StreamDEQ: 4 unroll steps, 5-frame lookback
# python tools/test.py configs/UR-StreamDEQ/mdeq_mpii_vid_256x256_4u_5f.py ${CKPT} --eval PCKh

# SUR-StreamDEQ: trained with stochastic unrolling, evaluated with a UR-StreamDEQ config
# python tools/test.py configs/UR-StreamDEQ/mdeq_mpii_vid_256x256_4u_5f.py ${CKPT} --eval PCKh
