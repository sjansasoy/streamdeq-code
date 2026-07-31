"""Extract z*_{t-1} for the real frame immediately preceding each Cityscapes validation image.

Parses data/list/cityscapes/val_sequence_2_frame_seq.lst (groups of 3 lines: 2 unlabeled context
frames from leftImg8bit_sequence/ + 1 labeled evaluation frame), and for each group takes the
second line -- the real frame immediately before the labeled one -- as "frame t-1". Computes its
baseline equilibrium (f_thres=27, cold start) the same way extract_convergence_mse.py does for the
labeled frames, and caches it under results/ml_project/z_star_cache/frame_t_minus_1/.

This is the population-level extension of the single-image pilot in
results/ml_project/fusion_contamination.ipynb (Section 16), and a prerequisite for
extract_fusion_contamination_realprev.py.

Run from MDEQ-Vision/:
    conda run -n streamdeq_city python scripts/ml_project/extract_zstar_prev_frame.py [--limit N]

Resumable: previous frames already cached are skipped.
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import time
from pathlib import Path

import cv2
import torch

MDEQ_VISION_ROOT = Path(__file__).resolve().parents[2]
os.chdir(MDEQ_VISION_ROOT)
sys.path.insert(0, "lib")

import models
import datasets
from config import config

CFG_PATH = "experiments/cityscapes/seg_mdeq_XL_sf_27i_gpu0.yaml"
SEQ_LIST_PATH = Path("data/list/cityscapes/val_sequence_2_frame_seq.lst")
DATA_ROOT = Path("data/cityscapes")
Z_STAR_PREV_DIR = Path("results/ml_project/z_star_cache/frame_t_minus_1")


def build_model_and_dataset():
    config.defrost()
    config.merge_from_file(CFG_PATH)
    config.freeze()

    model = eval("models." + config.MODEL.NAME + ".get_seg_net")(config)
    pretrained_dict = torch.load(config.TEST.MODEL_FILE, map_location="cpu", weights_only=False)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    print(f"loaded {len(pretrained_dict)} / {len(model_dict)} tensors from checkpoint")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda().eval()

    # Only used for its input_transform (mean/std normalization) -- the list_path/contents
    # of this dataset object are irrelevant here, we never call __getitem__ on it.
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset = eval("datasets." + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1,
    )
    return model, dataset


def get_prev_frame_paths():
    lines = [l for l in SEQ_LIST_PATH.read_text().splitlines() if l.strip()]
    assert len(lines) % 3 == 0, f"expected groups of 3 lines, got {len(lines)} total"
    prev_paths = []
    for i in range(0, len(lines), 3):
        prev_line = lines[i + 1]  # 2nd of the 2 context frames -- immediately precedes the labeled frame
        prev_paths.append(prev_line.strip())
    return prev_paths


def main(limit=None):
    model, dataset = build_model_and_dataset()
    prev_frame_paths = get_prev_frame_paths()
    total = len(prev_frame_paths) if limit is None else min(limit, len(prev_frame_paths))

    Z_STAR_PREV_DIR.mkdir(parents=True, exist_ok=True)
    processed_this_run = 0
    already_done = 0

    for rel_path in prev_frame_paths[:total]:
        frame_name = os.path.splitext(os.path.basename(rel_path))[0]
        z_star_path = Z_STAR_PREV_DIR / f"{frame_name}_f{config.DEQ.F_THRES}_z_star.pt"
        if z_star_path.exists():
            already_done += 1
            continue

        t0 = time.time()
        image_bgr = cv2.imread(str(DATA_ROOT / rel_path), cv2.IMREAD_COLOR)
        assert image_bgr is not None, f"failed to read {DATA_ROOT / rel_path}"
        image = dataset.input_transform(image_bgr)
        image = image.transpose((2, 0, 1))
        image_gpu = torch.from_numpy(image.copy()).unsqueeze(0).float().cuda()

        with torch.no_grad():
            z_star, _, _ = model._forward(
                [image_gpu, None], train_step=-1, compute_jac_loss=False, f_thres=config.DEQ.F_THRES,
            )
        z_star_cpu = [z.cpu() for z in z_star]
        torch.save(z_star_cpu, z_star_path)

        processed_this_run += 1
        print(f"[{already_done + processed_this_run}/{total}] {frame_name}  ({time.time() - t0:.1f}s)")

    print(f"done. {already_done + processed_this_run} / {total} previous frames cached "
          f"({processed_this_run} computed this run, {already_done} already cached).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most N previous frames (for smoke tests)")
    args = parser.parse_args()
    main(limit=args.limit)
