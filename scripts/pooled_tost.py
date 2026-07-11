"""Pooled equivalence test across three training seeds.

Per-seed TOST is underpowered on Freiburg (n=1,018). This averages each image's
outcome across the three seeds before bootstrapping, which uses every run without
inflating the effective sample size. Reproduces the pooled rows of Table 4.

Usage:
    python scripts/pooled_tost.py --datasets gsd freiburg
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

import timm
from eval_all import (
    CKPT_DIR,
    conf_correct,
    load_dataset,
)


def load_taus(results_csv):
    """Read the ARC-D tau chosen on the val split, per (dataset, seed)."""
    taus = {}
    if not os.path.exists(results_csv):
        return taus
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row["system"] == "ARC-D" and row["condition"] == "clean" and row["tau"]:
                taus[(row["dataset"], int(row["seed"]))] = float(row["tau"])
    return taus


def bootstrap_ci(diff, n_boot=5000, ci_lo=5, ci_hi=95):
    idx = np.random.randint(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(means, ci_lo), np.percentile(means, ci_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gsd", "freiburg"])
    ap.add_argument("--results", default="results/final_results.csv")
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    taus = load_taus(args.results)
    if not taus:
        print(f"no ARC-D taus found in {args.results} — run eval_all.py first")
        return

    conditions = [("clean", "none", 0), ("jpeg_s2", "jpeg", 2),
                  ("blur_s2", "blur", 2), ("occ_s1", "occlusion", 1)]

    for dataset in args.datasets:
        _val, test_items, ncls = load_dataset(dataset)
        print(f"\n===== {dataset} pooled equivalence test (n={len(test_items)}) =====")

        per_condition = {tag: {"arcd": [], "heavy": []} for tag, _, _ in conditions}

        for seed in (0, 1, 2):
            if (dataset, seed) not in taus:
                print(f"  seed {seed}: no tau, skipping")
                continue
            tau = taus[(dataset, seed)]

            distilled = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
            distilled.load_state_dict(torch.load(
                f"{CKPT_DIR}/{dataset}_distilled_seed{seed}.pt", map_location=device))
            heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
            heavy.load_state_dict(torch.load(
                f"{CKPT_DIR}/{dataset}_heavy_seed{seed}.pt", map_location=device))

            for tag, kind, sev in conditions:
                cd, kd = conf_correct(distilled, test_items, kind, sev, device)
                _, kh = conf_correct(heavy, test_items, kind, sev, device)
                keep = cd >= tau
                karcd = np.where(keep, kd, kh)
                per_condition[tag]["arcd"].append(karcd.astype(float))
                per_condition[tag]["heavy"].append(kh.astype(float))

            del distilled, heavy
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  seed {seed} done (tau={tau:.3f})")

        for tag, _, _ in conditions:
            if not per_condition[tag]["arcd"]:
                continue
            arcd_mean = np.mean(per_condition[tag]["arcd"], axis=0)  # per-image mean over seeds
            heavy_mean = np.mean(per_condition[tag]["heavy"], axis=0)
            mean_diff, lo, hi = bootstrap_ci(arcd_mean - heavy_mean)
            equivalent = lo > -args.margin and hi < args.margin
            verdict = "EQUIVALENT" if equivalent else "not established"
            print(f"  {tag:8s} pooled ARC-D vs heavy: "
                  f"{mean_diff * 100:+.2f}% "
                  f"[{lo * 100:+.2f}, {hi * 100:+.2f}]  {verdict}")


if __name__ == "__main__":
    main()
