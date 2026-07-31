# Cityscapes Experiments

This part of the repository handles video semantic segmentation experiments. Most of the code here is from the [original DEQ repo](https://github.com/locuslab/deq). For additional background on MDEQ, see the original DEQ repository.

## Installation

Run all commands below from `Cityscapes/MDEQ-Vision/`.

```shell
cd Cityscapes/MDEQ-Vision
pip install -r requirements.txt
```

## Dataset

The Cityscapes dataset is usually used with single-image training. For video evaluation, we use the sequence split `leftImg8bit_sequence_trainvaltest.zip`. You can download the dataset with [CityscapesScripts](https://github.com/mcordts/cityscapesScripts).

For streaming evaluation, we provide dataset lists in `data/list/cityscapes/` (for example, `val_sequence_5_frame.lst`).

## Usage

1. Download the pretrained MDEQ segmentation model from [here](https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view) and place it at the path set in `TEST.MODEL_FILE` inside each config.

2. Configs live under `experiments/cityscapes/` and are organized model-wise into `baseline/`, `IL-StreamDEQ/`, and `UR-StreamDEQ/`. UR- and SUR-StreamDEQ share the `UR-StreamDEQ/` folder since their streaming inference is identical; SUR training configs carry a `_stoch` suffix (for example, `seg_mdeq_XL_unroll_20_stoch.yaml`).

Run IL-StreamDEQ on streaming videos:

```shell
python tools/seg_test.py --cfg experiments/cityscapes/IL-StreamDEQ/seg_mdeq_XL_[NUM_FRAMES]f_[NUM_ITERS]i.yaml
```

Here `[NUM_FRAMES]` is the temporal lookback (matched by `DATASET.TEST_SET`, e.g. `val_sequence_5_frame.lst`) and `[NUM_ITERS]` is the number of solver iterations per frame. Baseline configs use the `sf` (single-frame) naming. UR-StreamDEQ streaming configs are named `seg_mdeq_XL_unroll_[NUM_FRAMES]f_[NUM_ITERS]i.yaml`.

See `scripts/` for additional training and streaming evaluation examples.

## Machine Learning project delivery

For the final Machine Learning project on scale-aware initialization for StreamDEQ, see:

`MDEQ-Vision/README_ML_PROJECT.md`

That file explains the modified code, required datasets/checkpoints, experiment YAMLs, reproduction scripts, and expected results.

Note that this project uses its own streaming configs and list files (the `_gpu0_seq` / `_seq` naming, plus everything under `experiments/cityscapes/ml_project/`) rather than upstream's `IL-StreamDEQ/` configs — see `README_ML_PROJECT.md` for why upstream's original list files cannot be used as shipped for streaming evaluation.
