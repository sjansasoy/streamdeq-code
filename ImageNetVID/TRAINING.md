# Training StreamDEQ on ImageNet-VID

This is a direct, no-detours guide to setting up and launching training for
this repo. It only covers the steps that actually work. For the full story
of *why* each step is needed — including the dead ends we ruled out along
the way — see [`SETUP_LOG.md`](SETUP_LOG.md) (sections 9-16 cover training
specifically).

## Why this differs from `README.md`

The README's instructions are outdated in a few places (e.g. the `mmcv-full`
wheel URL it links no longer has version 1.2.7 for PyTorch 1.10.0). This
guide reflects what we actually verified works, as of this writing, on an
Ubuntu 22.04 server with NVIDIA A30/L40S GPUs.

## Requirements

- **PyTorch 1.10.0**, not 1.7.0. Training (specifically, the backward pass
  through MDEQ's implicit-differentiation hook) segfaults reliably on
  PyTorch 1.7.0, regardless of GPU. This is an autograd stability issue in
  that PyTorch version, not a bug in this codebase. PyTorch 1.10.0 is also
  what the original authors used (confirmed by the official checkpoint's
  saved metadata). See `SETUP_LOG.md` section 16 for the full diagnosis.
- **`mmcv-full==1.2.7` built from source** against PyTorch 1.10.0 — no
  prebuilt wheel exists for that combination anymore.
- A CUDA-capable GPU. Evaluation runs fine on any recent NVIDIA GPU; training
  has only been validated on Ampere (A30/A100). It has *not* been validated
  on Ada Lovelace (L40S) — evaluation works there, but we haven't re-tested
  the training path on that architecture under PyTorch 1.10.0.
- The ImageNet-VID + ImageNet-DET dataset, and the ImageNet-pretrained MDEQ
  backbone checkpoint (see below).

## 1. Set up the environment

```bash
cd ImageNetVID
bash scripts/setup_train_env.sh [env_name]   # default env name: streamdeq_vid_train
```

This creates a conda environment with PyTorch 1.10.0, builds `mmcv-full`
1.2.7 and `mmpycocotools` from source, installs this repo as an editable
package, and restores `mmdet/models/backbones/lib/` (a folder that exists
outside of git — see `SETUP_LOG.md` section 2).

The script prints a verification step at the end (imports the backbone and
detector). If it completes without error, the environment is ready.

**If the build fails on a different machine:** the script was written for
Ubuntu 22.04. The most fragile steps are installing a matching host compiler
for `nvcc` and locating `crypt.h` (steps 3-5 in the script). Read the actual
compiler error rather than assuming this script's fixes apply verbatim —
see `SETUP_LOG.md` section 16.3 for the reasoning behind each fix, which
should transfer even if the exact commands don't.

## 2. Place the dataset

The training config combines two datasets. Both need COCO-VID-style JSON
annotations (not raw ILSVRC XML) — see `SETUP_LOG.md` section 11 for where
to get these if you don't already have them converted.

```
ImageNetVID/data/ILSVRC/
├── annotations/
│   ├── imagenet_vid_train.json
│   └── imagenet_det_30plus1cls.json
└── Data/
    ├── VID/       (from the ILSVRC2015 VID release)
    └── DET/train/ (from the ILSVRC2017 DET release — only the train split
                    is needed; ImageNet DET's val/test aren't referenced by
                    imagenet_det_30plus1cls.json)
```

Symlinks work fine here (`data/` is gitignored) — no need to copy hundreds
of GB of images into the repo directory.

## 3. Get the pretrained backbone

Download the ImageNet-pretrained MDEQ backbone (link in `README.md`,
Google Drive) to `pretrained_models/mdeq_XL_cls_new.pkl`:

```bash
pip install gdown  # if not already available
gdown "https://drive.google.com/uc?id=1vcWnlb5yUXE-3VHiuVJdfRZVeyx0U9-W" \
    -O pretrained_models/mdeq_XL_cls_new.pkl
```

## 4. Launch training

```bash
bash scripts/launch_training.sh <num_gpus> [env_name] [extra tools/train.py args...]

# e.g. on a 4-GPU machine:
bash scripts/launch_training.sh 4

# resuming an interrupted run:
bash scripts/launch_training.sh 4 streamdeq_vid_train \
    --resume-from work_dirs/faster_rcnn_mdeq_fpn_1x_imagenetvid/latest.pth
```

This runs `tools/dist_train.sh` against
`configs/baseline/faster_rcnn_mdeq_fpn_1x_imagenetvid.py` — the
non-streaming `MDEQ_FasterRCNN` config (7 epochs, matching the paper).
`StreamDEQ` (the streaming detector class used for evaluation) is not used
during training — `forward_train` isn't implemented for it.

Checkpoints save every epoch to `work_dirs/faster_rcnn_mdeq_fpn_1x_imagenetvid/`,
so an interrupted run is always resumable.

## Expected timing

Measured on 1× NVIDIA A30, `samples_per_gpu=1`, `F_THRES=26` (26 Broyden
iterations per image in both the forward and backward pass — matching the
paper's setting): **~2.3s/iteration**.

The combined VID+DET training set is ~1.47M images/epoch, over 7 epochs.
That works out to **months**, not days, on a small number of GPUs — this is
consistent with the paper's own acknowledgment that implicit-layer models
are expensive to train relative to explicit architectures, not a sign
something is misconfigured. Scale the GPU count accordingly; with N GPUs the
wall-clock time divides roughly by N (`dist_train.sh` splits the batch
across GPUs, one image per GPU per step).

## Evaluating a trained checkpoint

Use the same `tools/test.py` workflow documented in `SETUP_LOG.md` (sections
7-13), pointing `--checkpoint` at whatever `work_dirs/.../epoch_N.pth` (or
`latest.pth`) you want to evaluate, with one of the `configs/IL-StreamDEQ/*_stream_*`
configs.
