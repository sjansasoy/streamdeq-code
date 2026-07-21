# MPII Experiments

This part of the repository handles video pose estimation experiments.

## Installation

The code presented here merges [MMPose](https://github.com/open-mmlab/mmpose) with the [official implementation of MDEQ](https://github.com/locuslab/deq). In addition to MMPose, you will need the dependencies of the MDEQ repo.

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n streamdeq python=3.8 -y
    conda activate streamdeq
    ```

2. Install PyTorch (1.10.0) and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

3. Install mmcv-full, then install StreamDEQ (pose) in development mode.

    ```shell
    pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
    pip install -r requirements.txt
    python setup.py develop
    ```

## Dataset

We build a video variant of MPII for streaming evaluation. Structure the data paths following the standard [MMPose MPII layout](https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html#mpii). For video evaluation, the `TopDownMpiiVidDataset` samples the previous frames of each sequence.

## Usage

Configs are organized model-wise: `baseline/`, `IL-StreamDEQ/`, and `UR-StreamDEQ/` (MDEQ backbone definitions live in `configs/mdeq/`). UR- and SUR-StreamDEQ share the `UR-StreamDEQ/` folder since their streaming inference is identical; the SUR (stochastically-unrolled) configs carry a `_stoch` suffix.

### Training (static images)

```
python tools/train.py configs/baseline/mdeq_mpii_256x256_lr_1e4.py
```

UR-StreamDEQ (unrolled) and SUR-StreamDEQ (stochastically-unrolled) variants:

```
python tools/train.py configs/UR-StreamDEQ/mdeq_mpii_256x256_lr_1e4_unroll_20.py
python tools/train.py configs/UR-StreamDEQ/mdeq_mpii_256x256_lr_1e4_unroll_20_stoch.py
```

### Streaming Evaluation

Streaming configs follow the `mdeq_mpii_vid_256x256_{N}{i|u}_{F}f.py` pattern, where `F` is the temporal lookback (`num_prev_frames`), `{N}i` is the number of equilibrium-solver iterations per frame (IL-StreamDEQ), and `{N}u` is the number of unroll steps per frame (UR-StreamDEQ). SUR-StreamDEQ checkpoints are evaluated with a UR-StreamDEQ config.

```
# IL-StreamDEQ: 1 solver iteration, 5-frame lookback
python tools/test.py configs/IL-StreamDEQ/mdeq_mpii_vid_256x256_1i_5f.py path/to/model.pth --eval PCKh

# UR-StreamDEQ: 4 unroll steps, 5-frame lookback
python tools/test.py configs/UR-StreamDEQ/mdeq_mpii_vid_256x256_4u_5f.py path/to/model.pth --eval PCKh
```

See `scripts/eval_streaming.sh` for a runnable example.
