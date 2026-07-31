#!/usr/bin/env python
"""
Generate the full Fase 1 sweep of YAML configs: every context (1f/2f/5f/10f/20f)
x every scale-drop combination (0,1,2,3,4 scales dropped) x every Broyden
iteration budget (1i/2i/4i/8i).

Existing hand-written YAMLs (2f/5f, produced in earlier sessions) are left
untouched: this script only writes a file if it does not already exist, so it
is safe to re-run at any point to fill in whatever is still missing.
"""
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_ROOT = os.path.join(REPO_DIR, "experiments", "cityscapes", "ml_project")

CONTEXTS = [1, 2, 5, 10, 20]
ITERS = [1, 2, 4, 8]

# Folder name on disk vs. the "phase" label written to results_master.csv can
# differ: the existing master CSV (from earlier sessions) used "phase0_global"
# for the folder "phase0_global_init". Keep the folder name (files already
# live there) but log under the label already used historically, so the
# master CSV never ends up with two spellings of the same phase.
PHASE_LABELS = {
    "phase0_global_init": "phase0_global",
}


def phase_label(phase_dir):
    return PHASE_LABELS.get(phase_dir, phase_dir)


# (phase_dir, filename_token, DEQ.INIT_MODE, DEQ.PARTIAL_INIT_MODE or None, DEQ.STALE_K or None)
GLOBAL_MODES = [
    ("phase0_global_init", "previous", "previous", None, None),
    ("phase0_global_init", "zero", "zero", None, None),
    ("phase0_global_init", "stale_k2", "stale", None, 2),
]

# drop_count -> list of (phase_dir, filename_token, partial_init_mode)
PARTIAL_MODES = {
    1: [
        ("phase1_leave_one_scale", "drop_scale0_previous", "drop_scale0_previous"),
        ("phase1_leave_one_scale", "drop_scale1_previous", "drop_scale1_previous"),
        ("phase1_leave_one_scale", "drop_scale2_previous", "drop_scale2_previous"),
        ("phase1_leave_one_scale", "drop_scale3_previous", "drop_scale3_previous"),
    ],
    2: [
        # Filenames follow the convention already on disk for 2f/5f: the two
        # "adjacent block" pairs keep the legacy coarse/fine alias names
        # internally even though the filename is scaleX_scaleY_previous.
        ("phase1_two_scale", "scale0_scale1_previous", "fine_previous_coarse_zero"),
        ("phase1_two_scale", "scale0_scale2_previous", "scale0_scale2_previous"),
        ("phase1_two_scale", "scale0_scale3_previous", "scale0_scale3_previous"),
        ("phase1_two_scale", "scale1_scale2_previous", "scale1_scale2_previous"),
        ("phase1_two_scale", "scale1_scale3_previous", "scale1_scale3_previous"),
        ("phase1_two_scale", "scale2_scale3_previous", "coarse_previous_fine_zero"),
    ],
    3: [
        ("phase1_three_scale", "only_scale0_previous", "only_high_resolution_previous"),
        ("phase1_three_scale", "only_scale1_previous", "only_scale1_previous"),
        ("phase1_three_scale", "only_scale2_previous", "only_scale2_previous"),
        ("phase1_three_scale", "only_scale3_previous", "only_low_resolution_previous"),
    ],
}

TEMPLATE = """GPUS: (0,)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
WORKERS: 4
PRINT_FREQ: 100

MODEL:
  NAME: mdeq
  NUM_LAYERS: 3
  NUM_GROUPS: 8
  DROPOUT: 0.04
  WNORM: true
  DOWNSAMPLE_TIMES: 2
  EXPANSION_FACTOR: 6
  EXTRA:
    FINAL_CONV_KERNEL: 1
    FULL_STAGE:
      NUM_MODULES: 1
      NUM_BRANCHES: 4
      BLOCK: BASIC
      BIG_KERNELS:
        - 0
        - 0
        - 0
        - 0
      NUM_BLOCKS:
        - 1
        - 1
        - 1
        - 1
      NUM_CHANNELS:
        - 88
        - 176
        - 352
        - 704
      FUSE_METHOD: SUM
DEQ:
  F_SOLVER: 'broyden'
  B_SOLVER: 'broyden'
  STOP_MODE: 'abs'
  F_THRES: {f_thres}
  B_THRES: 30
  SPECTRAL_RADIUS_MODE: false
  MODE: 'stream'
  INIT_MODE: '{init_mode}'
{partial_line}{stale_line}CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
LOSS:
  USE_OHEM: false
  OHEMTHRES: 0.9
  OHEMKEEP: 131072
DATASET:
  DATASET: cityscapes
  ROOT: 'data/'
  TEST_SET: 'list/cityscapes/val_sequence_{context}_frame_seq.lst'
  TRAIN_SET: 'list/cityscapes/train.lst'
  NUM_CLASSES: 19
TRAIN:
  IMAGE_SIZE:
    - 1024
    - 768
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 2
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 500
  RESUME: true
  OPTIMIZER: sgd
  LR: 0.01
  WD: 0.0003
  MOMENTUM: 0.9
  NESTEROV: false
  FLIP: true
  MULTI_SCALE: true
  LR_SCHEDULER: 'cosine'
  DOWNSAMPLERATE: 1
  PRETRAIN_STEPS: 40000
  IGNORE_LABEL: 255
  SCALE_FACTOR: 16
TEST:
  IMAGE_SIZE:
    - 2048
    - 1024
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 1
  FLIP_TEST: false
  MULTI_SCALE: false
  MODEL_FILE: 'pretrained_models/MDEQ_XL_Seg.pkl'
"""


def render(context, f_thres, init_mode, partial_init_mode, stale_k):
    partial_line = f"  PARTIAL_INIT_MODE: '{partial_init_mode}'\n" if partial_init_mode else ""
    stale_line = f"  STALE_K: {stale_k}\n" if stale_k is not None else ""
    return TEMPLATE.format(
        f_thres=f_thres,
        init_mode=init_mode,
        partial_line=partial_line,
        stale_line=stale_line,
        context=context,
    )


def main():
    written, skipped = 0, 0
    manifest = []  # (yaml_relpath, phase, context, init_name, init_mode, partial_init_mode, iters)

    for context in CONTEXTS:
        for iters in ITERS:
            combos = []
            for phase_dir, token, init_mode, partial_mode, stale_k in GLOBAL_MODES:
                combos.append((phase_dir, token, init_mode, partial_mode, stale_k))
            for drop_count, modes in PARTIAL_MODES.items():
                for phase_dir, token, partial_mode in modes:
                    combos.append((phase_dir, token, "partial", partial_mode, None))

            for phase_dir, token, init_mode, partial_mode, stale_k in combos:
                out_dir = os.path.join(EXP_ROOT, phase_dir)
                os.makedirs(out_dir, exist_ok=True)
                fname = f"{context}f_{token}_{iters}i.yaml"
                fpath = os.path.join(out_dir, fname)
                rel = os.path.relpath(fpath, REPO_DIR)

                manifest.append((rel, phase_label(phase_dir), f"{context}f", token, init_mode, partial_mode or "", iters))

                if os.path.exists(fpath):
                    skipped += 1
                    continue

                content = render(context, iters, init_mode, partial_mode, stale_k)
                with open(fpath, "w") as f:
                    f.write(content)
                written += 1

    manifest_path = os.path.join(EXP_ROOT, "full_sweep_manifest.csv")
    with open(manifest_path, "w") as f:
        f.write("yaml_path,phase,context,init_name,init_mode,partial_init_mode,broyden_iterations\n")
        for row in manifest:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"YAML configs written: {written}")
    print(f"YAML configs already present (skipped): {skipped}")
    print(f"Total configs in full matrix: {len(manifest)}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
