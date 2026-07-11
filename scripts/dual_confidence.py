"""Evaluate the conference paper's dual-confidence mechanism as an evaluated baseline.

Runs both the cheap and heavy classifiers on every test image and keeps the higher
confidence prediction. Fixed cost of g_cheap + g_heavy = 4.89 GMACs per image (both
models always run). Reproduces the Dual-confidence [3] row in Tables 2 and 3.

Usage:
    python scripts/dual_confidence.py --datasets gsd freiburg --seeds 0 1 2
"""
import argparse
import csv
import os

import numpy as np
import torch

import timm
from eval_all import (
    CKPT_DIR,
    conf_correct,
    load_dataset,
)
from fvcore.nn import FlopCountAnalysis
from pooled_tost import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gsd", "freiburg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/dual_confidence_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = [("clean", "none", 0), ("jpeg_s2", "jpeg", 2),
                  ("blur_s2", "blur", 2), ("occ_s1", "occlusion", 1)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["dataset", "seed", "system", "condition", "acc", "gmacs"]
    new = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()

        for dataset in args.datasets:
            _val, test_items, ncls = load_dataset(dataset)

            dummy = torch.randn(1, 3, 224, 224).to(device)
            g_cheap = FlopCountAnalysis(
                timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device),
                dummy,
            ).total() / 1e9
            g_heavy = FlopCountAnalysis(
                timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device),
                dummy,
            ).total() / 1e9
            g_dual = g_cheap + g_heavy  # both models always run

            print(f"\n===== {dataset} dual-confidence baseline, {g_dual:.2f} GMACs =====")

            agg = {tag: [] for tag, _, _ in conditions}
            pooled = {tag: {"dual": [], "heavy": []} for tag, _, _ in conditions}

            for seed in args.seeds:
                cheap = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
                cheap.load_state_dict(torch.load(
                    f"{CKPT_DIR}/{dataset}_cheap_seed{seed}.pt", map_location=device))
                heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
                heavy.load_state_dict(torch.load(
                    f"{CKPT_DIR}/{dataset}_heavy_seed{seed}.pt", map_location=device))

                for tag, kind, sev in conditions:
                    cc, kc = conf_correct(cheap, test_items, kind, sev, device)
                    ch, kh = conf_correct(heavy, test_items, kind, sev, device)
                    # dual-confidence: keep whichever model reports higher max softmax
                    k_dual = np.where(cc >= ch, kc, kh)
                    agg[tag].append(k_dual.mean())
                    pooled[tag]["dual"].append(k_dual.astype(float))
                    pooled[tag]["heavy"].append(kh.astype(float))
                    w.writerow(dict(dataset=dataset, seed=seed, system="Dual-confidence",
                                    condition=tag, acc=k_dual.mean(), gmacs=g_dual))

                del cheap, heavy
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"  seed {seed} done")

            for tag, _, _ in conditions:
                a = np.array(agg[tag]) * 100
                d_mean = np.mean(pooled[tag]["dual"], axis=0)
                h_mean = np.mean(pooled[tag]["heavy"], axis=0)
                m, lo, hi = bootstrap_ci(d_mean - h_mean)
                print(f"  {tag:8s} acc={a.mean():5.1f}±{a.std():3.1f}%   "
                      f"vs heavy: {m * 100:+.2f}% [{lo * 100:+.2f}, {hi * 100:+.2f}]")

    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
