#!/usr/bin/env python
"""
Drive run_and_log.py over many rows of full_sweep_manifest.csv in one process,
so a whole context (or several) can be left running unattended in tmux.

Usage examples:
    python scripts/ml_project/run_sweep_batch.py --context 1f --gpu 1
    python scripts/ml_project/run_sweep_batch.py --context 1f,10f --gpu 1 --iters 1,2,4
    python scripts/ml_project/run_sweep_batch.py --phase phase1_three_scale --gpu 2

Continues past individual failures (logs a warning and moves on) so one bad
config does not stop an unattended multi-hour run. Already-logged rows in
results_master.csv are skipped automatically (handled inside run_and_log.py).
"""
import argparse
import csv
import os
import subprocess
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(REPO_DIR, "experiments", "cityscapes", "ml_project", "full_sweep_manifest.csv")
RUNNER = os.path.join(os.path.dirname(__file__), "run_and_log.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--context", help="Comma-separated, e.g. 1f,2f,5f (default: all)")
    ap.add_argument("--phase", help="Comma-separated phase dir names (default: all)")
    ap.add_argument("--iters", help="Comma-separated iteration budgets, e.g. 1,2,4,8 (default: all)")
    args = ap.parse_args()

    contexts = set(args.context.split(",")) if args.context else None
    phases = set(args.phase.split(",")) if args.phase else None
    iters = set(args.iters.split(",")) if args.iters else None

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    selected = [
        r for r in rows
        if (contexts is None or r["context"] in contexts)
        and (phases is None or r["phase"] in phases)
        and (iters is None or r["broyden_iterations"] in iters)
    ]

    print(f"Selected {len(selected)} configs out of {len(rows)} in manifest.")
    failures = []

    for i, row in enumerate(selected, 1):
        print(f"\n{'='*70}\n[{i}/{len(selected)}] {row['yaml_path']}\n{'='*70}")
        cmd = ["python", RUNNER, "--cfg", row["yaml_path"], "--gpu", str(args.gpu)]
        result = subprocess.run(cmd, cwd=REPO_DIR)
        if result.returncode != 0:
            print(f"[warn] failed (exit {result.returncode}), continuing: {row['yaml_path']}", file=sys.stderr)
            failures.append(row["yaml_path"])

    print(f"\nBatch finished. {len(selected) - len(failures)}/{len(selected)} succeeded.")
    if failures:
        print("Failed configs:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
