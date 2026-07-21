# Representation Recycling for Streaming Video Analysis

This is the official implementation of [Streaming Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2204.13492v3) (ECCV 2022) and its extended journal version, [Representation Recycling for Streaming Video Analysis](https://arxiv.org/abs/2204.13492) (Neurocomputing 2026). Please visit our [project page](https://ufukertenli.github.io/streamdeq/) for more detailed information and qualitative results.

## Abstract

StreamDEQ infers frame-wise representations on videos with minimal per-frame computation. In contrast to conventional methods where compute time grows at least linearly with the network depth, we update the representations in a continuous manner. We leverage implicit (deep equilibrium) layers, which infer the representation of an image by solving a fixed-point problem. Our main insight is to exploit the slowly changing nature of videos and use the previous frame representation as the initial condition on each frame. This scheme recycles recent inference computations and greatly reduces the needed processing time. StreamDEQ recovers near-optimal representations in a few frames and maintains an up-to-date representation throughout the video, achieving on par accuracy with the baseline MDEQ while being several times faster.

## Repository structure

Each task lives in its own folder with its own `requirements.txt`, configs, and `scripts/`:

| Task | Folder | Framework |
|------|--------|-----------|
| Video semantic segmentation | [`Cityscapes/`](Cityscapes/README.md) | MDEQ-Vision |
| Video object detection | [`ImageNetVID/`](ImageNetVID/README.md) | MMDetection + MDEQ |
| Video pose estimation | [`MPII/`](MPII/README.md) | MMPose + MDEQ |

Segmentation configs and scripts live under `Cityscapes/MDEQ-Vision/`; detection and pose use `configs/` and `scripts/` at the task root.

## Model variants

Training is performed on static images; streaming is an inference-only technique. Each task's configs are organized model-wise into `baseline/`, `IL-StreamDEQ/`, and `UR-StreamDEQ/`:

- **Baseline (MDEQ):** standard multiscale deep equilibrium model, solved to a fixed point per frame.
- **IL-StreamDEQ:** the implicit-layer model run in streaming mode, using the Broyden solver.
- **UR-StreamDEQ:** replaces the fixed-point solver with a fixed number of unrolling steps.
- **SUR-StreamDEQ:** a training-time option that unrolls to a random depth for improved robustness across iteration budgets. Its streaming inference is identical to UR-StreamDEQ, so it shares the `UR-StreamDEQ/` folder and is identified by a `_stoch` suffix in the config name.

## Qualitative Results

Baseline with 2 iterations per frame            |  IL-StreamDEQ with 2 iterations per frame
:-------------------------:|:-------------------------:
![](resources/Baseline_2_iteration.gif)  |  ![](resources/StreamDEQ_2_iteration.gif)

UR-StreamDEQ with 2 unroll steps per frame      |  SUR-StreamDEQ with 2 unroll steps per frame
:-------------------------:|:-------------------------:
![](resources/UR-StreamDEQ_2_unroll.gif)  |  ![](resources/SUR-StreamDEQ_2_unroll.gif)

The UR/SUR pair uses a different Cityscapes sequence than the baseline/IL pair above.

## Temporal lookback

How far back in a video StreamDEQ recycles representations is set per task via config:

- **Segmentation:** `DATASET.TEST_SET` (`val_sequence_{N}_frame.lst`) + iterations in the `seg_mdeq_XL_{N}f_{M}i.yaml` config name.
- **Detection:** the `{N}f` (reference frames) and `{M}i` (iterations) fields in the `..._stream_{N}f_{M}i.py` config name.
- **Pose:** `num_prev_frames` (`{N}f`) on `TopDownMpiiVidDataset` + iterations/unroll steps (`{N}i` / `{N}u`) in the config name.

## Quickstart

See each task's `README.md` for dataset setup and pretrained models, and `scripts/` for plain run examples (training and streaming evaluation).

Detection:
```shell
cd ImageNetVID
bash scripts/train.sh           # train (baseline / UR / SUR)
bash scripts/eval_streaming.sh  # streaming evaluation (IL / UR / SUR)
```

Segmentation:

```shell
cd Cityscapes/MDEQ-Vision
bash scripts/eval_streaming.sh  # streaming evaluation (IL / UR / SUR)
```

Pose:

```shell
cd MPII
bash scripts/eval_streaming.sh  # streaming evaluation (IL / UR / SUR)
```

## Contributing

Feel free to open an issue or a pull request at any time.

## Citation

If you find this repository useful, please consider citing our work:

```
@InProceedings{ertenli2022streaming,
  author="Ertenli, Can Ufuk and Akbas, Emre and Cinbis, Ramazan Gokberk",
  title="Streaming Multiscale Deep Equilibrium Models",
  booktitle="European Conference on Computer Vision (ECCV)",
  year="2022",
  pages="189--205",
  organization={Springer}
}

@article{ertenli2026representation,
  title={Representation recycling for streaming video analysis},
  author={Ertenli, Can Ufuk and Cinbis, Ramazan Gokberk and Akbas, Emre},
  journal={Neurocomputing},
  pages={134366},
  year={2026},
  publisher={Elsevier}
}
```
