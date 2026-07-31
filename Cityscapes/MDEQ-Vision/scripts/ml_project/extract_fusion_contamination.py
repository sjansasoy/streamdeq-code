"""Extract fusion-contamination measurements for the Cityscapes validation set.

For each validation image, reuses its already-cached z* (from
scripts/ml_project/extract_convergence_mse.py, under results/ml_project/z_star_cache/frame_t/)
as a stand-in for a "perfect previous frame". For each of the 4 "drop_scaleK_previous"
modes, builds the real hybrid initial state via the model's own
_build_partial_init_state() (scale K zeroed, the other 3 already at z*), re-solves at
f_thres=1..8 (the iteration budgets used in the real streaming sweep), and records the
resulting per-channel MSE against z* for all 4 scales. This measures how much resetting
one scale disturbs the other three once they're fused back together -- see
results/ml_project/fusion_contamination.ipynb for the single-image version this extends.

Run from MDEQ-Vision/:
    conda run -n streamdeq_city python scripts/ml_project/extract_fusion_contamination.py [--limit N]

Resumable: images already present in the results file are skipped. Images without a
cached z* (i.e. extract_convergence_mse.py hasn't processed them yet) are skipped with
a warning, not treated as an error.
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # free A30 at the time this was written

import sys
import time
from pathlib import Path

import torch

MDEQ_VISION_ROOT = Path(__file__).resolve().parents[2]
os.chdir(MDEQ_VISION_ROOT)
sys.path.insert(0, "lib")

import models
import datasets
from config import config

CFG_PATH = "experiments/cityscapes/seg_mdeq_XL_sf_27i_gpu0.yaml"
Z_STAR_DIR = Path("results/ml_project/z_star_cache/frame_t")
RESULTS_PATH = Path("results/ml_project/fusion_contamination_all_images.pt")

DROP_MODES = ["drop_scale0_previous", "drop_scale1_previous", "drop_scale2_previous", "drop_scale3_previous"]
F_THRES_VALUES = list(range(1, 9))  # 1..8, matching the iteration budgets used in the real streaming sweep


def build_model_and_loader():
    config.defrost()
    config.merge_from_file(CFG_PATH)
    config.freeze()

    model = eval("models." + config.MODEL.NAME + ".get_seg_net")(config)
    pretrained_dict = torch.load(config.TEST.MODEL_FILE, map_location="cpu", weights_only=False)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda().eval()

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval("datasets." + config.DATASET.DATASET)(
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
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return model, testloader


def load_existing_results():
    if RESULTS_PATH.exists():
        return torch.load(RESULTS_PATH, weights_only=False)
    return {"image_names": [], "mse": []}  # mse[i]: tensor (4 modes, 8 f_thres, 1320 channels)


def main(limit=None):
    model, testloader = build_model_and_loader()

    results = load_existing_results()
    done = set(results["image_names"])
    total = len(testloader) if limit is None else min(limit, len(testloader))
    print(f"{len(done)} / {total} images already processed")

    processed_this_run = 0
    for image, _label, _size, name in testloader:
        if limit is not None and processed_this_run >= limit:
            break
        image_name = name[0]
        if image_name in done:
            continue

        z_star_path = Z_STAR_DIR / f"{image_name}_f{config.DEQ.F_THRES}_z_star.pt"
        if not z_star_path.exists():
            print(f"[skip] no cached z* for {image_name}", file=sys.stderr)
            continue

        t0 = time.time()
        image_gpu = image.cuda()
        z_star_cpu = torch.load(z_star_path, map_location="cpu", weights_only=False)
        z_star = [z.cuda() for z in z_star_cpu]
        total_channels = sum(z.shape[1] for z in z_star)

        mse_all = torch.zeros(len(DROP_MODES), len(F_THRES_VALUES), total_channels)
        with torch.no_grad():
            for mode_idx, mode in enumerate(DROP_MODES):
                init_state = model._build_partial_init_state(z_star, mode)
                for ft_idx, ft in enumerate(F_THRES_VALUES):
                    y, _, _ = model._forward(
                        [image_gpu, init_state], train_step=-1, compute_jac_loss=False, f_thres=ft,
                    )
                    per_channel_mse = torch.cat([
                        (y[i] - z_star[i]).pow(2).mean(dim=(0, 2, 3)).cpu() for i in range(4)
                    ])
                    mse_all[mode_idx, ft_idx] = per_channel_mse

        results["image_names"].append(image_name)
        results["mse"].append(mse_all)
        torch.save(results, RESULTS_PATH)

        processed_this_run += 1
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{len(results['image_names'])}/{total}] {image_name}  ({time.time() - t0:.1f}s)  "
              f"| GPU allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")

    print(f"done. {len(results['image_names'])} / {total} images in {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most N new images (for smoke tests)")
    args = parser.parse_args()
    main(limit=args.limit)
