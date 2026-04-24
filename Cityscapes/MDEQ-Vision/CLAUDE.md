# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamDEQ is a semantic segmentation framework for video inference on Cityscapes. It uses Multiscale Deep Equilibrium Networks (MDEQ) to perform fixed-point iterations across video frames, enabling streaming inference that reuses previous frame representations to achieve 3x+ speedup over single-frame baselines.

## Commands

All scripts are run from the `tools/` directory:

```bash
cd tools/

# Training
python seg_train.py --cfg ../experiments/cityscapes/seg_mdeq_XL_1f_2i_gpu0_seq.yaml

# Evaluation
python seg_test.py --cfg ../experiments/cityscapes/seg_mdeq_XL_1f_2i_gpu0_seq.yaml

# Override config values on the command line
python seg_train.py --cfg config.yaml TRAIN.LR 0.005 TRAIN.END_EPOCH 100

# Optional flags
#   --modelDir   override output directory
#   --logDir     override log directory
#   --dataDir    override data directory
#   --testModel  path to weights for evaluation
#   --percent    fraction of training data to use (0–1)
#   --local_rank GPU rank for distributed training
```

Experiment configs follow the naming convention `seg_mdeq_XL_{N}f_{K}i_gpu0_seq.yaml` where `N` = number of frames in the sequence and `K` = number of DEQ iterations.

## Architecture

### Config System (`lib/config/`)
YACS-based hierarchical config. Defaults are in `default.py`; YAML files in `experiments/` override them; CLI key=value pairs override YAML. Key config sections: `MODEL`, `DEQ`, `LOSS`, `DATASET`, `TRAIN`, `TEST`.

### Model (`lib/models/`)
- `mdeq.py` — `get_seg_net(config)` factory; creates `MDEQSegNet`
- `mdeq_core.py` — `MDEQNet` core: 3–4 multi-scale branches run through a Broyden fixed-point solver; `BasicBlock` is the residual unit (optional 5×5 kernels, weight norm)

### Streaming vs. Baseline
`DEQ.MODE = 'stream'` enables streaming: fixed-point iterations reuse the previous frame's equilibrium state as a warm start. `DEQ.MODE = 'baseline'` restarts from zero each frame. The frame count in the config filename (1f, 2f, 5f, 10f, 20f) controls how many frames are fed per sequence; iteration count (1i, 2i, 4i, 8i) controls DEQ solver steps.

### Dataset (`lib/datasets/`)
`Cityscapes` inherits `BaseDataset`. Data lists are `.lst` files under `data/list/cityscapes/`. Streaming evaluation uses `val_sequence_*_frame_seq.lst` files; training uses `train.lst`. Labels use 19 classes; label 255 is ignored.

### Training Loop (`lib/core/seg_function.py`)
`train()` runs per-epoch; `validate()` computes mIoU. Distributed training via `DistributedDataParallel` + `DistributedSampler`. Only rank-0 saves checkpoints and logs. Two-phase training: optional Jacobian regularization for first `PRETRAIN_STEPS` steps, then main phase.

### Loss (`lib/core/seg_criterion.py`)
`CrossEntropy` (class-weighted) or `OhemCrossEntropy` (online hard example mining, controlled by `LOSS.USE_OHEM`).

### Utilities (`lib/utils/`)
- `FullModel` wraps model + loss for multi-GPU loss distribution
- `get_optimizer()` builds SGD or Adam from config
- `create_logger()` writes timestamped logs to `log/` and stdout

## Checkpoints
- `output/<experiment>/checkpoint.pth.tar` — latest epoch (used for `TRAIN.RESUME = true`)
- `output/<experiment>/best.pth` — best mIoU
- `output/<experiment>/final_state.pth` — end of training
- State dict keys prefixed with `"model."` are stripped automatically when loading

## Pretrained Models
Place pretrained weights in `pretrained_models/`. The XL segmentation backbone expects `mdeq_XL_cls.pkl`; the full seg model expects `MDEQ_XL_Seg.pkl`.
