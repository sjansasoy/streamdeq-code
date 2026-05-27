# Cityscapes Experiments

This part of the repository handles video semantic segmentation experiments. Most of the code here is from the original DEQ repo:

https://github.com/locuslab/deq

For installation instructions and detailed explanations about the original code, please visit the original DEQ repo.

## Dataset

The Cityscapes dataset is usually used with single-image training. However, the official website provides snippets from which the train, val, and test sets are created. The dataset split used for video evaluation is `leftImg8bit_sequence_trainvaltest.zip`.

You can download the dataset using the CityscapesScripts repo:

https://github.com/mcordts/cityscapesScripts

For video evaluation, dataset lists are provided in the `data/list/` folder.

## Usage

1. Download the pretrained MDEQ segmentation model from:

   https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view

2. Run the following command to test StreamDEQ on streaming videos:

       python tools/seg_test.py --cfg experiments/cityscapes/seg_MDEQ_XL_[NUM_FRAMES]f_[NUM_ITERS]i.yaml

   where `[NUM_FRAMES]` is the length of each video and `[NUM_ITERS]` is the number of iterations performed per frame.

   Note that `sf` in the config names refers to the single-frame baseline cases. Those configs are equivalent to the config files in the original MDEQ.

## Machine Learning project delivery

For the final Machine Learning project on scale-aware initialization for StreamDEQ, see:

`MDEQ-Vision/README_ML_PROJECT.md`

That file explains the modified code, required datasets/checkpoints, experiment YAMLs, reproduction scripts, and expected results.
