#!/usr/bin/env python
"""
Run a single seg_test.py config, parse its final mIoU/PixelAcc/MeanAcc line,
and append a row to results/ml_project/results_master.csv.

Usage:
    python scripts/ml_project/run_and_log.py --cfg <path/to/config.yaml> --gpu 1

The (phase, context, init_name, init_mode, partial_init_mode, broyden_iterations)
metadata for the row is looked up from full_sweep_manifest.csv by matching the
yaml path, so it does not need to be re-derived from the filename.

Safe to re-run: if a row for this exact (phase, context, init_name,
broyden_iterations) already exists in results_master.csv, the run is skipped
unless --force is passed.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_ROOT = os.path.join(REPO_DIR, "experiments", "cityscapes", "ml_project")
MANIFEST_PATH = os.path.join(EXP_ROOT, "full_sweep_manifest.csv")
RESULTS_CSV = os.path.join(REPO_DIR, "results", "ml_project", "results_master.csv")
RESULTS_FIELDS = ["phase", "context", "init_name", "init_mode", "partial_init_mode", "broyden_iterations", "miou", "notes"]

MIOU_RE = re.compile(
    r"MeanIU:\s*([0-9.]+),\s*Pixel_Acc:\s*([0-9.]+),\s*Mean_Acc:\s*([0-9.]+)"
)


def load_manifest():
    rows = {}
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            rows[row["yaml_path"]] = row
    return rows


def already_done(phase, context, init_name, iters):
    if not os.path.exists(RESULTS_CSV):
        return False
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if (row["phase"] == phase and row["context"] == context
                    and row["init_name"] == init_name
                    and str(row["broyden_iterations"]) == str(iters)):
                return True
    return False


def append_result(phase, context, init_name, init_mode, partial_init_mode, iters, miou, notes=""):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "phase": phase,
            "context": context,
            "init_name": init_name,
            "init_mode": init_mode,
            "partial_init_mode": partial_init_mode,
            "broyden_iterations": iters,
            "miou": f"{miou:.4f}",
            "notes": notes,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to yaml config (repo-relative or absolute)")
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--force", action="store_true", help="Re-run even if already logged")
    args = ap.parse_args()

    cfg_abs = args.cfg if os.path.isabs(args.cfg) else os.path.join(REPO_DIR, args.cfg)
    cfg_rel = os.path.relpath(cfg_abs, REPO_DIR)

    manifest = load_manifest()
    if cfg_rel not in manifest:
        print(f"[skip] {cfg_rel} not found in full_sweep_manifest.csv, cannot log metadata", file=sys.stderr)
        sys.exit(1)
    meta = manifest[cfg_rel]

    if not args.force and already_done(meta["phase"], meta["context"], meta["init_name"], meta["broyden_iterations"]):
        print(f"[skip] already logged: {meta['phase']} {meta['context']} {meta['init_name']} {meta['broyden_iterations']}i")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = ["python", "tools/seg_test.py", "--cfg", cfg_rel]
    print(f"[run] GPU={args.gpu} {' '.join(cmd)}")
    start = time.time()

    proc = subprocess.Popen(
        cmd, cwd=REPO_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail_lines = []
    for line in proc.stdout:
        print(line, end="")
        tail_lines.append(line)
        if len(tail_lines) > 200:
            tail_lines.pop(0)
    proc.wait()
    elapsed_min = (time.time() - start) / 60.0

    if proc.returncode != 0:
        print(f"[error] seg_test.py exited with code {proc.returncode} for {cfg_rel}", file=sys.stderr)
        sys.exit(proc.returncode)

    match = None
    for line in reversed(tail_lines):
        m = MIOU_RE.search(line)
        if m:
            match = m
            break

    if match is None:
        print(f"[error] could not parse MeanIU from output for {cfg_rel}", file=sys.stderr)
        sys.exit(2)

    miou = float(match.group(1))
    notes = f"full_sweep_runner, {elapsed_min:.1f} min"
    append_result(meta["phase"], meta["context"], meta["init_name"], meta["init_mode"],
                  meta["partial_init_mode"], meta["broyden_iterations"], miou, notes)
    print(f"[done] {meta['context']} {meta['init_name']} {meta['broyden_iterations']}i -> mIoU={miou:.4f} ({elapsed_min:.1f} min)")


if __name__ == "__main__":
    main()
