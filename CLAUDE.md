# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the official implementation of [Streaming Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2204.13492) (StreamDEQ, ECCV 2022). StreamDEQ uses Multiscale Deep Equilibrium Networks (MDEQ) — implicit layers that infer representations via fixed-point (Broyden) iteration — and speeds up video inference by warm-starting each frame's fixed-point solve from the previous frame's equilibrium state, instead of solving from scratch every frame.

The repo contains **two independent, self-contained subprojects** that share the MDEQ/DEQ concept but have separate codebases, dependencies, and entry points. There is no shared build system between them.

- **`Cityscapes/MDEQ-Vision/`** — video semantic segmentation on Cityscapes. Custom PyTorch training/eval framework (based on the [locuslab/deq](https://github.com/locuslab/deq) repo) with a YACS config system. See **`Cityscapes/MDEQ-Vision/CLAUDE.md`** for full detail on this subproject's commands and architecture — read it before working in that directory.
- **`ImageNetVID/`** — video object detection on ImageNet VID. Built as a fork of [mmdetection](https://github.com/open-mmlab/mmdetection) (standard `mmdet` layout: `configs/`, `mmdet/`, `tools/`), with an added MDEQ backbone and a `StreamDEQ` detector.

Fork remotes: `origin` = `sjansasoy/streamdeq-code`, `upstream` = `ufukertenli/streamdeq-code` (original paper authors).

The current branch (`ml-project-delivery`) also carries a course ML project — **scale-aware initialization for StreamDEQ** — layered on top of the Cityscapes subproject. That project's full documentation (motivation, modified files, experiment configs, reproduction scripts, expected results) lives in `Cityscapes/MDEQ-Vision/README_ML_PROJECT.md`; read it before touching `INIT_MODE`/`PARTIAL_INIT_MODE`/`STALE_K` logic.

## Cityscapes/MDEQ-Vision (segmentation)

See `Cityscapes/MDEQ-Vision/CLAUDE.md` for commands and architecture. In short: scripts are run from `tools/` (`seg_train.py`, `seg_test.py`) against YACS YAML configs in `experiments/`, following the naming convention `seg_mdeq_XL_{N}f_{K}i_gpu0_seq.yaml` (N = frames per sequence, K = DEQ solver iterations). `DEQ.MODE` switches between `stream` (warm-started) and `baseline` (reset every frame).

The ML-project layer adds solver-initialization modes on top of this (`INIT_MODE=previous|zero|stale|partial`, `PARTIAL_INIT_MODE=<name>` to selectively reuse multiscale states) — see `README_ML_PROJECT.md` for the full mode/scale reference table and reproduction scripts under `scripts/ml_project/`.

## ImageNetVID (detection)

This subproject is a standard mmdetection checkout (`mmdet` version pinned in `mmdet/version.py`; MMCV version compatibility is listed in `docs/get_started.md`) plus StreamDEQ-specific additions:

- `mmdet/models/backbones/mdeq.py` — MDEQ backbone (`MDEQModule`, multi-branch fixed-point solver, mirrors the Cityscapes MDEQ core).
- `mmdet/models/detectors/mdeq_faster_rcnn.py`, `mmdet/models/detectors/streamdeq.py` — Faster R-CNN detectors built on the MDEQ backbone. `StreamDEQ.simple_test` implements the streaming warm-start: it walks the reference-frame buffer, feeding each frame's output as the next frame's fixed-point injection, and only trains on/predicts the final frame in a sequence (`forward_train` is deliberately `NotImplementedError` — **video training is not supported**, only streaming inference).
- Configs live in `configs/mdeq/` (single-frame baseline) and `configs/IL-StreamDEQ/` (streaming) and `configs/UR-StreamDEQ/` (unrolled variants), named `faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_{N}f_{K}i.py` (N = frames, K = iterations), extending the shared bases in `configs/_base_/`.

### Commands (run from `ImageNetVID/`)

```bash
# Single-GPU train / test (standard mmdetection CLI)
python tools/train.py configs/IL-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_5f_1i.py
python tools/test.py configs/IL-StreamDEQ/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_5f_1i.py <checkpoint> --eval bbox

# Multi-GPU (torch.distributed)
bash tools/dist_train.sh <config> <num_gpus>
bash tools/dist_test.sh <config> <checkpoint> <num_gpus>

# Slurm cluster variants
bash tools/slurm_train.sh <partition> <job_name> <config> <work_dir>
bash tools/slurm_test.sh <partition> <job_name> <config> <checkpoint>
```

Test with `pytest.ini` / `requirements/tests.txt`; linting via `.dev_scripts/linter.sh` and `.pre-commit-config.yaml` (both inherited from upstream mmdetection, not StreamDEQ-specific).

## Current work: reproducing StreamDEQ on ImageNet-VID

- Goal: reproduce the paper's ImageNet-VID results (`ImageNetVID/`), not the Cityscapes segmentation side.
- Training/evaluation runs on this server (136.145.54.118), which has 3 GPUs.
- The ImageNet-VID dataset lives on a separate server; access from this machine is not yet configured. Don't assume the dataset is locally reachable — check with the user before running data-dependent commands.
- Immediate next step: run evaluation with pretrained checkpoints (if the repo/paper provides any) to validate the pipeline end-to-end, before attempting training from scratch.
- The user is new to Python/PyTorch. When working in this repo, explain the code and underlying concepts (e.g. what a config does, what a checkpoint contains, what an eval script computes) as you go rather than just running commands.
