# Scale-aware initialization for StreamDEQ on Cityscapes

This repository contains the code used for the Machine Learning final project:

**Scale-aware and quality-aware initialization for StreamDEQ in semantic segmentation on Cityscapes.**

The project studies how different initialization strategies for the Broyden solver affect MDEQ/StreamDEQ inference on Cityscapes. The main question is whether the temporal benefit of StreamDEQ depends equally on all internal MDEQ scales, or whether some scales are more important for warm-starting the solver.

This document is meant to be sufficient, on its own, for someone else to set up the environment, obtain the data, and reproduce every experiment reported here — including the parts of the setup that are easy to get wrong (in particular, the dataset path issue described in [Dataset requirements](#dataset-requirements)).

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

## Environment setup

There is no `requirements.txt` or `environment.yml` checked into this repository, so the working environment is documented here instead. The experiments in this project were run with a conda environment named `streamdeq_city`, with the following key package versions:

```text
Python        3.8.20
torch         2.4.1  (with CUDA)
opencv-python 4.13.0
numpy         1.23.5
yacs
```

For the analysis notebooks (`graphs.ipynb`, `results/ml_project/analisis_resultados.ipynb`, and the convergence/fusion notebooks described in the sections below), the following additional packages are required:

```text
pandas
seaborn      0.13.2
matplotlib
scipy                  (only needed by equilibrium_convergence.ipynb, for scipy.stats.spearmanr)
nbconvert    7.16.6   (only needed to execute notebooks non-interactively, e.g. via CI or a headless run)
ipykernel    6.29.5   (only needed to execute notebooks non-interactively)
```

To install the notebook dependencies into an existing environment:

```bash
conda activate streamdeq_city
pip install seaborn nbconvert ipykernel
```

The rest of the base dependencies (PyTorch, OpenCV, YACS, etc.) follow the same requirements as the original DEQ/MDEQ-Vision repo linked above; that repo's own setup instructions are the authoritative source for those.

Everything here is inference-only. Training is not part of this project and was not exercised, so training-only dependencies were not verified.

---

## Dataset requirements

Cityscapes is **not included** in this repository due to size and license restrictions.

For this project, two Cityscapes components are required:

1. The standard Cityscapes train/val/test data, which provides:
   - `gtFine/`
   - `leftImg8bit/`

2. The Cityscapes sequence data, which provides:
   - `leftImg8bit_sequence/`

The semantic labels are taken from `gtFine/`, and the video-frame inputs are taken from `leftImg8bit_sequence/` (see the important note below about why `leftImg8bit/` alone is not enough). Therefore, reproducing the streaming experiments requires **both** the standard Cityscapes data (`leftImg8bit_sequence_trainvaltest.zip`'s sibling, the standard `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`) and the sequence data (`leftImg8bit_sequence_trainvaltest.zip`). Both are available from the official Cityscapes download page and can be fetched with the [CityscapesScripts](https://github.com/mcordts/cityscapesScripts) repo.

To reproduce the experiments, download Cityscapes from the official source and place or symlink the required folders under:

```text
data/cityscapes/
├── gtFine/
├── leftImg8bit/
└── leftImg8bit_sequence/
```

### Important: why the `_seq` list files exist (and why you cannot just reuse upstream's own lists)

The upstream repo already ships its own streaming-eval list files (`data/list/cityscapes/val_sequence_{N}_frame.lst`, without the `_seq` suffix) and its own experiment configs (`experiments/cityscapes/seg_mdeq_XL_{N}f_{K}i.yaml`, without the `_gpu0_seq` suffix). **Those original list files are broken for streaming evaluation as shipped.**

Each line in a `val_sequence_{N}_frame.lst` file is one frame to feed through the model; lines without a second (label) column are unlabeled "priming" frames used to warm up the streaming state before the one labeled frame per sequence is evaluated. In upstream's original lists, priming frames are pointed at:

```text
leftImg8bit/val/<city>/<...>_leftImg8bit.png
```

But `leftImg8bit/val/` only contains the ~500 officially annotated Cityscapes validation frames (e.g. **267** files for `frankfurt`), not the neighboring frames of each video snippet. The actual neighboring frames only exist under:

```text
leftImg8bit_sequence/val/<city>/<...>_leftImg8bit.png
```

(e.g. **8010** files for `frankfurt` — every frame of every snippet, roughly 30x more). Running upstream's own configs/lists as-is against a real Cityscapes install fails with `FileNotFoundError` on the priming frames, because those files simply do not exist under `leftImg8bit/val/`.

The fix applied in this repo is a new set of list files, `data/list/cityscapes/val_sequence_{N}_frame_seq.lst`, which are otherwise identical to upstream's own lists except that every **unlabeled priming-frame** path is rewritten from `leftImg8bit/...` to `leftImg8bit_sequence/...`. The one **labeled** line per sample (the frame that is actually scored) is left pointing at `leftImg8bit/...` + `gtFine/...`, since that frame genuinely is one of the officially annotated ones and does exist there. All experiment YAMLs in this project (`*_gpu0_seq.yaml` and everything under `experiments/cityscapes/ml_project/`) point `DATASET.TEST_SET` at these `_seq` lists, never at upstream's originals.

`_seq` list files exist for all five context lengths used in this project:

```text
data/list/cityscapes/
├── val_sequence_1_frame_seq.lst
├── val_sequence_2_frame_seq.lst
├── val_sequence_5_frame_seq.lst
├── val_sequence_10_frame_seq.lst
└── val_sequence_20_frame_seq.lst
```

All five always evaluate the same 500 validation images (Cityscapes val) with mIoU; what changes between them is how many extra priming frames precede each evaluated frame (1, 2, 5, 10, or 20 requested — see the note below on 20f). More priming frames means the streaming state has had more steps to settle before the ablation (dropping/keeping scales) is applied on the scored frame, which is itself one of the axes this project studies.

**Note on `20f`:** `val_sequence_20_frame_seq.lst` has 10,000 lines rather than the 10,500 you'd get from a clean `500 × 21`. This is expected, not a bug: a handful of validation frames near the start of their raw Cityscapes video snippet do not have a full 20 preceding frames available, so those samples get truncated to fewer priming frames. All five list files still score the same 500 labeled frames; only the *available* amount of priming near sequence boundaries differs slightly for `10f`/`20f`.

**There is currently no script in this repository that generates these `_seq` list files.** They were produced once (by rewriting the priming-frame paths as described above) and are checked in directly. If you need to regenerate them (e.g. for a context length not already covered), replicate the same rule: keep every line from the corresponding upstream `val_sequence_{N}_frame.lst`, and rewrite the image path of every line that has no second (label) column from `leftImg8bit/...` to `leftImg8bit_sequence/...`.

---

## Pretrained checkpoint

The pretrained checkpoint is **not included** in this repository.

To reproduce the reported results, place the checkpoint at:

```text
pretrained_models/MDEQ_XL_Seg.pkl
```

Download link (from the original StreamDEQ repo): https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view

All experiments use:

```yaml
TEST:
  MODEL_FILE: 'pretrained_models/MDEQ_XL_Seg.pkl'
```

No retraining or fine-tuning is performed anywhere in this project — every experiment is a `tools/seg_test.py` evaluation run against this fixed checkpoint.

---

## Main initialization modes

The code adds support for the following initialization modes.

### Global initialization (`DEQ.INIT_MODE`)

- `INIT_MODE='previous'`: use the representation from the immediately previous frame. This corresponds to the original StreamDEQ warm-start behavior.
- `INIT_MODE='zero'`: initialize the solver from zero (no temporal warm-start).
- `INIT_MODE='stale'` with `STALE_K=2`: initialize using a temporally stale representation, corresponding to `z_{t-2}` instead of `z_{t-1}`. Falls back to `previous` automatically if fewer than `STALE_K` frames have been processed yet (e.g. near the start of a sequence).

### Partial multiscale initialization (`DEQ.INIT_MODE='partial'`)

- `INIT_MODE='partial'`
- `PARTIAL_INIT_MODE=<mode>` selects which scales are reused

In partial initialization, selected scales from `self.prev_outs` are reused and the remaining scales are reset to zero before being handed to the Broyden solver. Nothing else about the model changes: all four scale branches keep running and fusing every frame regardless of `PARTIAL_INIT_MODE` — only the *initial* value of the reset scale(s) is zero instead of the previous frame's equilibrium.

The diagnosed MDEQ scales are:

| Scale | Shape | Interpretation |
|---|---|---|
| 0 | `(1, 88, 256, 512)` | highest resolution / fine scale |
| 1 | `(1, 176, 128, 256)` | mid-high resolution |
| 2 | `(1, 352, 64, 128)` | mid-low resolution |
| 3 | `(1, 704, 32, 64)` | lowest resolution / coarse scale |

Full reference of every `PARTIAL_INIT_MODE` value implemented in `_build_partial_init_state` (`lib/models/mdeq.py`), grouped by how many scales are dropped:

| Scales dropped | `PARTIAL_INIT_MODE` | Scales reused |
|---|---|---|
| 0 | `all_previous` | `[0, 1, 2, 3]` |
| 1 | `drop_scale0_previous` | `[1, 2, 3]` (alias: `scale1_scale2_scale3_previous`) |
| 1 | `drop_scale1_previous` | `[0, 2, 3]` |
| 1 | `drop_scale2_previous` | `[0, 1, 3]` |
| 1 | `drop_scale3_previous` | `[0, 1, 2]` |
| 2 | `fine_previous_coarse_zero` | `[0, 1]` |
| 2 | `coarse_previous_fine_zero` | `[2, 3]` |
| 2 | `scale0_scale2_previous` | `[0, 2]` |
| 2 | `scale0_scale3_previous` | `[0, 3]` |
| 2 | `scale1_scale2_previous` | `[1, 2]` |
| 2 | `scale1_scale3_previous` | `[1, 3]` |
| 3 | `only_high_resolution_previous` | `[0]` |
| 3 | `only_scale1_previous` | `[1]` |
| 3 | `only_scale2_previous` | `[2]` |
| 3 | `only_low_resolution_previous` | `[3]` |
| 4 | `all_zero` | `[]` |

Passing any other string raises `ValueError` (fails fast rather than silently defaulting to something unintended).

Historical naming note: `fine_previous_coarse_zero` / `coarse_previous_fine_zero` are the original names for the `{0,1}` / `{2,3}` pairs from an earlier phase of the project (when scales were grouped as "fine" vs "coarse" blocks); later phases introduced the more explicit `scaleI_scaleJ_previous` naming for the other four pairs. Some YAML filenames under `phase1_two_scale/` are named `scale0_scale1_previous_*.yaml` / `scale2_scale3_previous_*.yaml` for consistency with the other pair filenames, even though internally they set `PARTIAL_INIT_MODE` to the older `fine_previous_coarse_zero` / `coarse_previous_fine_zero` strings — this is intentional and both names produce identical behavior.

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
| `lib/config/default.py` | Adds configuration variables `INIT_MODE`, `STALE_K`, and `PARTIAL_INIT_MODE`. |
| `lib/core/seg_function.py` | Reads those config variables (with safe `getattr` defaults, so old configs that don't set them keep the original behavior) and passes them through `validate()`, `testval()`, and `test()` into `multi_scale_inference()`. |
| `lib/datasets/cityscapes.py` | Forwards `init_mode`, `stale_k`, and `partial_init_mode` from `multi_scale_inference()` through `inference()` into the model's `forward()` call, for both the plain and flipped-image inference paths. |
| `lib/models/mdeq.py` | Implements the actual selection logic: `_select_init_state()` dispatches on `init_mode`, and `_build_partial_init_state()` builds the per-scale reuse/zero state for `init_mode='partial'`. Also adds `self.prev_outs_history` (a short rolling buffer, capped at 10 entries) so `init_mode='stale'` can look back `STALE_K` frames; `reset_prev_outs()` clears both `prev_outs` and `prev_outs_history` between video sequences. |

### Known minor implementation details (not bugs, but worth knowing about)

- `lib/models/mdeq.py`'s `MDEQSegNet.forward()` still contains a one-time debug print block (guarded by `self._printed_prevouts_diagnostic`) that dumps the shape/dtype/device of each scale the first time streaming mode runs. It was used to diagnose the scale table above and was never removed. It only prints once per process and does not affect results, but it is dead debugging code.
- `lib/core/seg_function.py` has a stray `from logging import config` import at module level. It does nothing harmful — every function in the file that uses `config` receives it as its own local parameter, which shadows the module-level import — but it serves no purpose and was most likely an accidental IDE autocomplete import.

---

## Repository layout

```text
experiments/cityscapes/ml_project/
├── phase0_global_init/       # INIT_MODE in {previous, zero, stale} — 60 configs
├── phase1_leave_one_scale/    # drop exactly 1 scale (4 variants)     — 80 configs
├── phase1_two_scale/          # keep exactly 2 scales (6 variants)    — 120 configs
└── phase1_three_scale/        # keep exactly 1 scale (4 variants)     — 80 configs

scripts/ml_project/
├── run_phase0_global_init.sh        # original, hand-written reproduction scripts
├── run_phase1_leave_one_scale.sh    # (2f/5f only, partial iteration coverage —
├── run_phase1_two_scale.sh          #  see "Original phase 0 / phase 1 scripts" below)
├── generate_full_sweep_configs.py   # generates every {context}x{scales-dropped}x{iters} YAML
├── run_and_log.py                   # runs one YAML, parses mIoU, appends a row to results_master.csv
├── run_sweep_batch.py               # drives run_and_log.py over many YAMLs, skips already-logged rows
├── extract_convergence_mse.py       # per-scale/per-channel convergence toward z*, all 500 images
├── extract_fusion_contamination.py  # fusion-contamination measurement, idealized same-image "previous frame", all 500
├── extract_zstar_prev_frame.py      # z*_(t-1) for the real preceding video frame, all 500
└── extract_fusion_contamination_realprev.py  # fusion-contamination measurement, real previous frame, all 500

results/ml_project/
├── results_master.csv                  # one row per (phase, context, init_name, broyden_iterations) run
├── analisis_resultados.ipynb           # report-ready charts for the main sweep, in English
├── equilibrium_convergence.ipynb       # per-scale/per-channel convergence toward z*, one image (baseline mode)
├── convergence_population_analysis.ipynb  # same, aggregated over all 500 validation images
├── fusion_contamination.ipynb          # does resetting one scale contaminate the others via fusion? idealized + real previous frame, one image + all 500
├── explicacion_equilibrium_convergence.md # Spanish write-up of the convergence notebooks, step by step
├── z_star_cache/                       # cached equilibria, *.pt, gitignored
│   ├── frame_t/                        #   labeled evaluation frames (~43 GB for all 500)
│   └── frame_t_minus_1/                #   real preceding video frames, all 500 (extract_zstar_prev_frame.py)
├── convergence_mse_all_images.pt       # output of extract_convergence_mse.py, gitignored (~70 MB)
├── fusion_contamination_all_images.pt  # output of extract_fusion_contamination.py, gitignored (~85 MB)
└── fusion_contamination_realprev_all_images.pt  # output of extract_fusion_contamination_realprev.py, gitignored (~105 MB)

data/list/cityscapes/
├── val_sequence_1_frame_seq.lst
├── val_sequence_2_frame_seq.lst
├── val_sequence_5_frame_seq.lst
├── val_sequence_10_frame_seq.lst
└── val_sequence_20_frame_seq.lst
```

Every combination of context length × scale configuration × Broyden iteration budget is tracked in a single manifest, generated by `generate_full_sweep_configs.py`:

```text
experiments/cityscapes/ml_project/full_sweep_manifest.csv
```

This is the full parameter space this project explores:

- **Contexts**: `1f, 2f, 5f, 10f, 20f` (how many priming frames before the scored frame — see the dataset section above)
- **Scale configurations**: 17 total — `previous`, `zero`, `stale_k2`, 4 leave-one-scale-out, 6 leave-two-scale-out (keep-2), 4 leave-three-scale-out (keep-1)
- **Broyden iterations**: `1, 2, 4, 8`

5 contexts × 17 configurations × 4 iteration budgets = **340 total runs**.

A separate `graphs.ipynb` notebook (at `MDEQ-Vision/graphs.ipynb`) also exists for ad-hoc/exploratory charting while the sweep is still running; it is intentionally excluded from version control (`.gitignore`'s `MDEQ-Vision/*.ipynb` rule) since it is meant to be edited freely without producing diff noise. `results/ml_project/analisis_resultados.ipynb` is the polished, report-oriented notebook and **is** tracked.

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

The final line of output (and of the log file under `output/cityscapes/<config_name>/`) reports the metrics for that run, e.g.:

```text
MeanIU:  0.2349, Pixel_Acc:  0.7683, Mean_Acc:  0.3437, Class IoU: [...]
```

---

## Original phase 0 / phase 1 scripts

These were the first reproduction scripts written for this project, and only cover `2f`/`5f` with partial iteration coverage (they predate the full sweep described below). From `Cityscapes/MDEQ-Vision`, run:

```bash
bash scripts/ml_project/run_phase0_global_init.sh
bash scripts/ml_project/run_phase1_leave_one_scale.sh
bash scripts/ml_project/run_phase1_two_scale.sh
```

To select a different GPU:

```bash
GPU_ID=1 bash scripts/ml_project/run_phase0_global_init.sh
```

These scripts only call `tools/seg_test.py` for each config in a hard-coded list; they do not write to `results_master.csv` themselves (the rows they produced were added by hand). Prefer the full-sweep tooling below for anything beyond re-running these specific historical configs.

---

## Running the full parameter sweep

The full 340-run matrix (all 5 contexts × 17 scale configurations × 4 iteration budgets) is generated and executed with three scripts under `scripts/ml_project/`, all run from `Cityscapes/MDEQ-Vision`:

### 1. Generate every YAML config

```bash
python scripts/ml_project/generate_full_sweep_configs.py
```

This writes any missing YAML under `experiments/cityscapes/ml_project/{phase0_global_init,phase1_leave_one_scale,phase1_two_scale,phase1_three_scale}/` and regenerates `experiments/cityscapes/ml_project/full_sweep_manifest.csv` (the full 340-row manifest used by the other two scripts). It never overwrites a YAML that already exists on disk, so it is always safe to re-run.

### 2. Run one config and log its result

```bash
python scripts/ml_project/run_and_log.py --cfg <path/to/config.yaml> --gpu <gpu_id>
```

Runs `tools/seg_test.py` for that one config, parses the final `MeanIU` line from its output, and appends a row to `results/ml_project/results_master.csv`. Looks up `phase`/`context`/`init_name`/`init_mode`/`partial_init_mode`/`broyden_iterations` for that row from `full_sweep_manifest.csv` by matching the YAML path — it does not re-derive them from the filename. If a row for that exact `(phase, context, init_name, broyden_iterations)` combination is already in `results_master.csv`, the run is skipped (pass `--force` to re-run anyway).

### 3. Run a whole batch (a context, a phase, a set of iteration budgets, or everything)

```bash
python scripts/ml_project/run_sweep_batch.py --gpu <gpu_id> [--context 1f,2f] [--phase phase1_two_scale] [--iters 1,2,4,8]
```

Filters `full_sweep_manifest.csv` by whichever of `--context`/`--phase`/`--iters` you pass (omit a filter to include everything), then calls `run_and_log.py` once per remaining row, in order, on the given GPU. It continues past individual failures (logs a warning and moves on) rather than aborting the whole run, since a multi-hour unattended sweep should not stop because one config crashed.

Example: run everything for context `10f` on GPU 1, leaving the run in a `tmux` session so it survives an SSH disconnect:

```bash
tmux new -s streamdeq_sweep_10f
conda activate streamdeq_city
cd Cityscapes/MDEQ-Vision
python scripts/ml_project/run_sweep_batch.py --context 10f --gpu 1 2>&1 | tee logs/full_sweep_10f.log
```

Multiple `run_sweep_batch.py` invocations can run concurrently on different GPUs (or even share one GPU, at the cost of each run taking longer due to compute contention) — `run_and_log.py`'s skip-if-already-logged check makes it safe to split the matrix across sessions this way.

To check overall progress at any point:

```python
import csv
manifest_total, done = {}, {}
for r in csv.DictReader(open("experiments/cityscapes/ml_project/full_sweep_manifest.csv")):
    manifest_total[r["context"]] = manifest_total.get(r["context"], 0) + 1
for r in csv.DictReader(open("results/ml_project/results_master.csv")):
    done[r["context"]] = done.get(r["context"], 0) + 1
for ctx in ["1f", "2f", "5f", "10f", "20f"]:
    print(ctx, done.get(ctx, 0), "/", manifest_total.get(ctx, 0))
```

---

## Analysis notebooks

Two notebooks consume `results/ml_project/results_master.csv` (the main sweep's results); both use the conda env described in [Environment setup](#environment-setup) (`streamdeq_city`, plus `seaborn`). Three more notebooks investigate follow-up questions beyond the main sweep — see [Follow-up investigation: does scale 1 converge slowest?](#follow-up-investigation-does-scale-1-converge-slowest-to-its-own-equilibrium) and [Follow-up investigation: fusion contamination](#follow-up-investigation-does-resetting-scale-1-contaminate-the-other-scales-through-fusion) below.

- **`MDEQ-Vision/graphs.ipynb`** — exploratory notebook, edited freely while the sweep is still filling in, not tracked by git.
- **`results/ml_project/analisis_resultados.ipynb`** — the report-ready notebook, tracked by git, entirely in English. Sections, in order: an organized progress table (configs run per context × iteration, out of 17 possible per cell); a shared color/line-style convention reused by every chart (each scale keeps the same identity color everywhere; `previous`/`zero` are neutral gray references; `stale_k2` is a dashed highlight); the "master curve" (all 17 configurations, one large figure per iteration budget); baselines (`previous`/`zero`/`stale_k2` only); a normalized "fraction of the zero→previous gap recovered" metric per configuration; leave-1/2/3-out curves; a summary curve averaged by how many scales are kept active; a best-pair-vs-worst-pair comparison; and a per-scale marginal-importance curve (average recovery with the scale active minus average recovery with it inactive, pooled across every other configuration).

To (re-)execute either notebook non-interactively and refresh its output with whatever is currently in `results_master.csv`:

```bash
conda activate streamdeq_city
cd Cityscapes/MDEQ-Vision   # or results/ml_project, for analisis_resultados.ipynb
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

---

## Expected results

The full results table is `results/ml_project/results_master.csv` (up to 340 rows as the sweep completes; see [Running the full parameter sweep](#running-the-full-parameter-sweep) for current progress). The main patterns observed, consistent across every context length tested so far (`1f` through `20f`) and across Broyden iteration budgets:

```text
previous > stale_k2 > zero
```

confirming that *any* temporal warm-start helps, especially at low iteration budgets — the gap between strategies shrinks as more Broyden iterations are allowed, since the solver has more chances to correct a poor initialization.

```text
previous ≈ drop_scale0_previous > drop_scale1_previous
```

i.e. dropping scale 0 (the finest/highest-resolution scale) barely hurts — sometimes it even matches or very slightly beats `previous` at low iteration counts — while dropping scale 1 hurts substantially. Scales 2 and 3 fall in between.

For two-scale combinations, the best configurations among those tested were:

```text
scale1_scale3_previous
scale1_scale2_previous
```

and the worst were the pairs that exclude scale 1 (`scale0_scale2_previous`, `scale0_scale3_previous`). Both best pairs include scale 1, and this holds at both `1i` and `2i` Broyden budgets — reinforcing that **scale 1 (the mid-high resolution scale) is consistently the most important scale for warm-starting the Broyden solver**, while scale 0 is consistently the least important. This is visible directly in `results/ml_project/analisis_resultados.ipynb`'s per-scale marginal-importance chart, computed by pooling across all 17 configurations rather than relying on any single leave-N-out group.

---

## Follow-up investigation: does scale 1 converge slowest to its own equilibrium?

The main sweep above establishes *that* scale 1 matters most for warm-starting, but not *why*. The most direct hypothesis is that scale 1 is simply the "hardest" of MDEQ's 4 branches to solve — the one that takes the Broyden solver longest to reach its own fixed point `z*`, independent of streaming altogether. Two notebooks test this, entirely in `DEQ.MODE=baseline` (no streaming, no `PARTIAL_INIT_MODE` — every scale starts from zero on every call):

- **`results/ml_project/equilibrium_convergence.ipynb`** — one validation image. Runs the solver to full convergence (`f_thres=27`) to obtain `z*`, caches it, then re-runs the solver at every budget `f_thres=1..27` and measures the MSE of each of MDEQ's 1320 individual channels (across all 4 scales) against `z*` at every step.
- **`results/ml_project/convergence_population_analysis.ipynb`** — the same measurement, aggregated across all 500 Cityscapes validation images. It does **not** load the model or touch the GPU itself; it only reads the file produced by:

  ```bash
  python scripts/ml_project/extract_convergence_mse.py [--limit N]
  ```

  run from `Cityscapes/MDEQ-Vision`. For every validation image, this script computes (or reuses, if already cached) that image's `z*` under `results/ml_project/z_star_cache/frame_t/<image_name>_f27_z_star.pt`, then re-runs the solver at every budget from 1 to 27 and records the resulting per-channel MSE to `results/ml_project/convergence_mse_all_images.pt`. It is resumable (already-processed images are skipped) and takes roughly 45 seconds per image — a few hours for all 500.

**Result:** scale 0 (finest resolution) converges fastest, scale 3 (coarsest) slowest, with scales 1 and 2 in between — **scale 1 is never the slowest scale to converge**, in either the single-image or the 500-image version. This rules out the simplest explanation for the main sweep's finding. See the notebooks' own closing summaries for the full breakdown (per-channel heterogeneity within each scale, the correlation between a channel's convergence speed and its influence on the final prediction, and whether a channel's convergence speed is a fixed trait or depends on image content).

---

## Follow-up investigation: does resetting scale 1 contaminate the other scales through fusion?

Since scale 1 isn't the slowest to converge on its own, a different mechanism was hypothesized: MDEQ fuses all 4 scales together at **every** solver iteration (`fullstage`, in `lib/models/mdeq_core.py`), so maybe resetting scale 1 disturbs the *other* 3 scales more than resetting any other scale does, once everything gets fused back together.

**`results/ml_project/fusion_contamination.ipynb`** tests this. Sections 1-7 use one validation image: for each of the 4 `drop_scaleK_previous` modes, it builds the actual hybrid initial state the model uses in streaming inference (`model._build_partial_init_state()`, the same method described in [Partial multiscale initialization](#partial-multiscale-initialization-deqinit_modepartial) above) — using that image's own cached `z*` as an idealized stand-in for "the previous frame", so scale `K` starts at zero while the other 3 already start at the correct answer — then re-solves for `f_thres=1..8` (the budgets used in the real streaming sweep) and measures how far each of the 4 scales ends up from `z*`. Sections 8-11 of the same notebook then repeat this measurement across all 500 validation images, reading from:

```bash
python scripts/ml_project/extract_fusion_contamination.py [--limit N]
```

run from `Cityscapes/MDEQ-Vision`. This script reuses each image's already-cached `z*` from `z_star_cache/frame_t/` rather than recomputing it, and writes per-channel MSE for all 4 `drop_scaleK` modes × 8 iteration budgets to `results/ml_project/fusion_contamination_all_images.pt`. Resumable, roughly 15-20 seconds per image (about 2-2.5 hours for all 500 — much faster than the convergence extraction above, since it never needs `f_thres` above 8). Sections 8-11 of the notebook only read that file (plus `z_star_cache/frame_t/`, to normalize) and do not need the model or the GPU.

**Result:** resetting scale 1 causes a clearly higher "contamination score" (average normalized MSE of the 3 *un*-reset scales) than resetting any other scale, both for the single image and across the full 500-image validation set (median score at `f_thres=4`: 0.22 for scale 0, **0.28 for scale 1**, 0.17 for scale 2, 0.12 for scale 3) — solid evidence in favor of the fusion hypothesis. This setup is still an idealization (a same-image stand-in for "the previous frame", not a genuinely earlier one) — the follow-up below drops that idealization.

### Does the same pattern hold with a real previous frame?

Sections 15-24 of the same notebook redo the whole experiment using the *real* frame immediately preceding each labeled validation image (Cityscapes ships these unlabeled frames under `leftImg8bit_sequence/`), instead of pretending each image is its own "previous frame". Two scripts (both under `scripts/ml_project/`, run from `Cityscapes/MDEQ-Vision`) support this:

```bash
python scripts/ml_project/extract_zstar_prev_frame.py [--limit N]
python scripts/ml_project/extract_fusion_contamination_realprev.py [--limit N]
```

`extract_zstar_prev_frame.py` computes `z*_{t-1}` (baseline, `f_thres=27`) for each of the 500 real preceding frames (found via `data/list/cityscapes/val_sequence_2_frame_seq.lst`), caching them under `z_star_cache/frame_t_minus_1/`; it must be run first (~3-4 seconds per frame, ~30-40 minutes for all 500 — one forward pass per image, not the 28-pass sweep `extract_convergence_mse.py` does, hence the large speed difference). `extract_fusion_contamination_realprev.py` then builds, for each image, 5 configurations from `z*_{t-1}` — `no_reset` (the full real warm start, nothing zeroed, i.e. the control) plus the 4 `drop_scaleK_previous` modes — re-solves at `f_thres=1..8`, and records per-channel MSE against the labeled frame's own `z*_t`, writing to `results/ml_project/fusion_contamination_realprev_all_images.pt` (~20-23 seconds per image, ~3 hours for all 500). Both scripts are resumable and were smoke-tested on 3 images (reproducing the single-image pilot's exact numbers) before running on all 500.

**Result:** the pattern holds, both for one image and across all 500. Contamination score at `f_thres=4`, median:

| | scale 0 | scale 1 | scale 2 | scale 3 |
|---|---|---|---|---|
| idealized, 500 images | 0.224 | **0.276** | 0.171 | 0.122 |
| real previous frame, 500 images | 0.299 | **0.378** | 0.278 | 0.215 |

Same ranking in both (reset scale 1 worst, reset scale 3 least bad); every real-frame score is higher than its idealized counterpart, as expected — real consecutive frames carry a small amount of genuine content drift the idealized same-image setup couldn't have, adding a baseline error on top of the pure fusion-contamination effect. The convergence-speed control (`no_reset`, population equivalent of the single-image Section 17) also reproduces the known ordering with a real warm start: median MSE reduction from `f_thres=1` to `8` is 86.8% for scale 0 down to only 26.2% for scale 3.

What remains open is the fully paper-faithful setup: `z*_{t-1}` here is still a clean 27-iteration solve for a single previous frame, not the partially-converged state a real multi-frame streaming chain with a small iteration budget would actually produce.

---

## Reproducibility notes

- Cityscapes is not included and must be downloaded separately (both the standard split and the sequence split — see [Dataset requirements](#dataset-requirements)).
- The pretrained checkpoint is not included and must be placed manually.
- No environment/requirements file is checked into the repo; see [Environment setup](#environment-setup) for the versions this project was run with.
- **Do not reuse upstream's own `val_sequence_{N}_frame.lst` list files or `seg_mdeq_XL_{N}f_{K}i.yaml` configs for streaming evaluation** — use the `_seq` list files and `_gpu0_seq` / `ml_project/` configs in this repo instead, for the reason explained in [Dataset requirements](#dataset-requirements).
- Experiments are inference-only. No retraining or fine-tuning is performed.
- The model architecture, weights, solver, segmentation head, dataset, and metrics are kept fixed. Only the initialization state passed to the Broyden solver is changed.
- `run_and_log.py` / `run_sweep_batch.py` are resumable: re-running them skips any `(phase, context, init_name, broyden_iterations)` combination already present in `results_master.csv`.
- `extract_convergence_mse.py`, `extract_fusion_contamination.py`, `extract_zstar_prev_frame.py`, and `extract_fusion_contamination_realprev.py` are all resumable: re-running any of them skips any image already present in its output file.
- `z_star_cache/`, `convergence_mse_all_images.pt`, `fusion_contamination_all_images.pt`, and `fusion_contamination_realprev_all_images.pt` (all under `results/ml_project/`) are **not tracked in git** — `*.pt` is gitignored repo-wide. If you clone this repo fresh, these do not exist yet; regenerate them, in order, by running `extract_convergence_mse.py` (produces `z_star_cache/frame_t/`), `extract_fusion_contamination.py` (reuses that cache), `extract_zstar_prev_frame.py` (produces `z_star_cache/frame_t_minus_1/`), then `extract_fusion_contamination_realprev.py` (reuses both caches) — all under `scripts/ml_project/`.
- `z_star_cache/` has two subfolders so it's always unambiguous which frame a cached equilibrium belongs to: `frame_t/` holds equilibria of labeled evaluation frames (written by `extract_convergence_mse.py` and by `equilibrium_convergence.ipynb`), and `frame_t_minus_1/` holds equilibria of real preceding video frames from `leftImg8bit_sequence/` (written by `extract_zstar_prev_frame.py` for all 500, and by `fusion_contamination.ipynb`'s Section 16 for the single pilot image).
