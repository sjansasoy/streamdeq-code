"""Extract z* and per-channel MSE-across-iterations for the Cityscapes validation set.

Baseline (non-streaming) DEQ mode only: for every validation image, runs a fully-converged
forward pass (f_thres=27) to get z*, then re-runs f_thres=1..27 comparing each result to z*.
This is the population-level extension of results/ml_project/equilibrium_convergence.ipynb.

Run from MDEQ-Vision/:
    conda run -n streamdeq_city python scripts/ml_project/extract_convergence_mse.py [--limit N]

Resumable: images whose z* is already cached under results/ml_project/z_star_cache/frame_t/ are
reused instead of recomputed, and the MSE results file is loaded and appended to (not overwritten)
if it already exists, so an interrupted run can just be restarted.

This script only ever computes z* for the labeled evaluation frame (frame t) -- z_star_cache/ also
has a frame_t_minus_1/ subfolder for real previous-frame equilibria, written by a different script
(see results/ml_project/fusion_contamination.ipynb, Section 16), kept separate so it's always clear
which frame a cached z* belongs to.
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # the 10f sweep GPU; deliberately not GPU 2 (20f)

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
MSE_RESULTS_PATH = Path("results/ml_project/convergence_mse_all_images.pt")


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
    if MSE_RESULTS_PATH.exists():
        return torch.load(MSE_RESULTS_PATH, weights_only=False)
    return {"image_names": [], "mse": []}  # mse[i]: tensor (27, 1320), concat order matches Section 6


def get_or_compute_z_star(model, image_gpu, image_name, f_thres):
    z_star_path = Z_STAR_DIR / f"{image_name}_f{f_thres}_z_star.pt"
    if z_star_path.exists():
        return torch.load(z_star_path, map_location=image_gpu.device, weights_only=False)

    with torch.no_grad():
        z_star, _, _ = model._forward(
            [image_gpu, None], train_step=-1, compute_jac_loss=False, f_thres=f_thres,
        )
    Z_STAR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save([z.detach().cpu() for z in z_star], z_star_path)
    return z_star


def main(limit=None):
    model, testloader = build_model_and_loader()
    f_thres_values = list(range(1, config.DEQ.F_THRES + 1))

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

        t0 = time.time()
        image_gpu = image.cuda()
        z_star = get_or_compute_z_star(model, image_gpu, image_name, config.DEQ.F_THRES)

        mse_per_scale = [[] for _ in range(4)]
        with torch.no_grad():
            for ft in f_thres_values:
                z_ft, _, _ = model._forward(
                    [image_gpu, None], train_step=-1, compute_jac_loss=False, f_thres=ft,
                )
                for i in range(4):
                    per_channel_mse = (z_ft[i] - z_star[i]).pow(2).mean(dim=(0, 2, 3)).cpu()
                    mse_per_scale[i].append(per_channel_mse)
        mse_all = torch.cat([torch.stack(s) for s in mse_per_scale], dim=1)  # (27, 1320)

        results["image_names"].append(image_name)
        results["mse"].append(mse_all)
        torch.save(results, MSE_RESULTS_PATH)

        processed_this_run += 1
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{len(results['image_names'])}/{total}] {image_name}  ({time.time() - t0:.1f}s)  "
              f"| GPU allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")

    print(f"done. {len(results['image_names'])} / {total} images in {MSE_RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="process at most N new images (for smoke tests)")
    args = parser.parse_args()
    main(limit=args.limit)
