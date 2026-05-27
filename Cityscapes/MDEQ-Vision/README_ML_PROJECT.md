# Scale-aware initialization for StreamDEQ on Cityscapes

This repository contains the code used for the Machine Learning final project:

**Scale-aware and quality-aware initialization for StreamDEQ in semantic segmentation on Cityscapes.**

The project studies how different initialization strategies for the Broyden solver affect MDEQ/StreamDEQ inference on Cityscapes. The main question is whether the temporal benefit of StreamDEQ depends equally on all internal MDEQ scales, or whether some scales are more important for warm-starting the solver.

---

## Code base and attribution

This project builds on the official StreamDEQ Cityscapes codebase:

```text
https://github.com/ufukertenli/streamdeq-code/tree/main/Cityscapes
```

The StreamDEQ implementation itself builds on the original DEQ/MDEQ codebase from LocusLab:

```text
https://github.com/locuslab/deq
```

The contribution of this project is not a new architecture or a new training procedure. Instead, this repository modifies the StreamDEQ/MDEQ-Vision inference pipeline to support controlled experiments on the initial state passed to the Broyden solver.

The model architecture, pretrained weights, Broyden solver, segmentation head, dataset, and evaluation metrics are kept fixed. Only the initialization state given to the solver during inference is changed.

---

## Main initialization modes

The code adds support for the following initialization modes:

### Global initialization

- `INIT_MODE='previous'`: use the representation from the immediately previous frame. This corresponds to the original StreamDEQ warm-start behavior.
- `INIT_MODE='zero'`: initialize the solver from zero.
- `INIT_MODE='stale'` with `STALE_K=2`: initialize using a temporally stale representation, corresponding to `z_{t-2}`.

### Partial multiscale initialization

- `INIT_MODE='partial'`
- `PARTIAL_INIT_MODE=<mode>`

In partial initialization, selected scales from `self.prev_outs` are reused and the remaining scales are reset to zero.

The diagnosed MDEQ scales are:

| Scale | Shape | Interpretation |
|---|---|---|
| 0 | `(1, 88, 256, 512)` | highest resolution / fine scale |
| 1 | `(1, 176, 128, 256)` | mid-high resolution |
| 2 | `(1, 352, 64, 128)` | mid-low resolution |
| 3 | `(1, 704, 32, 64)` | lowest resolution / coarse scale |

Examples:

| `PARTIAL_INIT_MODE` | Reused scales |
|---|---|
| `drop_scale0_previous` | `[1, 2, 3]` |
| `drop_scale1_previous` | `[0, 2, 3]` |
| `drop_scale2_previous` | `[0, 1, 3]` |
| `drop_scale3_previous` | `[0, 1, 2]` |
| `scale1_scale3_previous` | `[1, 3]` |
| `scale1_scale2_previous` | `[1, 2]` |
| `fine_previous_coarse_zero` | `[0, 1]` |
| `coarse_previous_fine_zero` | `[2, 3]` |

---

## Modified files

The main modified files are:

```text
lib/config/default.py
lib/core/seg_function.py
lib/datasets/cityscapes.py
lib/models/mdeq.py
```

Summary:

| File | Modification |
|---|---|
| `lib/config/default.py` | Adds configuration variables such as `INIT_MODE`, `STALE_K`, and `PARTIAL_INIT_MODE`. |
| `lib/core/seg_function.py` | Passes initialization options through the evaluation pipeline. |
| `lib/datasets/cityscapes.py` | Forwards `init_mode`, `stale_k`, and `partial_init_mode` into model inference. |
| `lib/models/mdeq.py` | Implements the logic for `zero`, `previous`, `stale`, and `partial` solver initialization. |

---

## Project files

The clean project files are organized as:

```text
experiments/cityscapes/ml_project/
├── phase0_global_init/
├── phase1_leave_one_scale/
└── phase1_two_scale/

scripts/ml_project/
├── run_phase0_global_init.sh
├── run_phase1_leave_one_scale.sh
└── run_phase1_two_scale.sh

results/ml_project/
└── results_master.csv
```

---

## Dataset requirements

Cityscapes is **not included** in this repository due to size and license restrictions.

To reproduce the experiments, download Cityscapes from the official source and place or symlink the required folders under:

```text
data/cityscapes/
├── gtFine/
├── leftImg8bit/
└── leftImg8bit_sequence/
```

The sequence list files used by the experiments are under:

```text
data/list/cityscapes/
├── val_sequence_2_frame_seq.lst
└── val_sequence_5_frame_seq.lst
```

The YAMLs use:

```yaml
DATASET:
  ROOT: 'data/'
  TEST_SET: 'list/cityscapes/val_sequence_2_frame_seq.lst'
```

or:

```yaml
DATASET:
  ROOT: 'data/'
  TEST_SET: 'list/cityscapes/val_sequence_5_frame_seq.lst'
```

---

## Pretrained checkpoint

The pretrained checkpoint is **not included** in this repository.

To reproduce the reported results, place the checkpoint at:

```text
pretrained_models/MDEQ_XL_Seg.pkl
```

All experiments use:

```yaml
MODEL:
  MODEL_FILE: 'pretrained_models/MDEQ_XL_Seg.pkl'
```

---

## How to run one experiment

From:

```text
Cityscapes/MDEQ-Vision
```

run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/seg_test.py \
    --cfg experiments/cityscapes/ml_project/phase0_global_init/2f_previous_1i.yaml
```

Another example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/seg_test.py \
    --cfg experiments/cityscapes/ml_project/phase1_leave_one_scale/2f_drop_scale1_previous_1i.yaml
```

---

## How to reproduce experiment groups

From:

```text
Cityscapes/MDEQ-Vision
```

run:

```bash
bash scripts/ml_project/run_phase0_global_init.sh
bash scripts/ml_project/run_phase1_leave_one_scale.sh
bash scripts/ml_project/run_phase1_two_scale.sh
```

To select a different GPU:

```bash
GPU_ID=1 bash scripts/ml_project/run_phase0_global_init.sh
```

---

## Expected results

The expected results reported in the project report and presentation are summarized in:

```text
results/ml_project/results_master.csv
```

Main observed patterns:

```text
previous > stale_k2 > zero
```

and:

```text
previous ≈ drop_scale0_previous > drop_scale1_previous
```

For two-scale combinations, the best configurations among those tested were:

```text
scale1_scale3_previous
scale1_scale2_previous
```

Both include scale 1, suggesting that scale 1 is particularly important for warm-starting the solver under the evaluated configurations.

---

## Reproducibility notes

- Cityscapes is not included and must be downloaded separately.
- The pretrained checkpoint is not included and must be placed manually.
- Experiments are inference-only.
- No retraining or fine-tuning is performed.
- The model architecture, weights, solver, segmentation head, dataset, and metrics are kept fixed.
- Only the initialization state passed to the Broyden solver is changed.
