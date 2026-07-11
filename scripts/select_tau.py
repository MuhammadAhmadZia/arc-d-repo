"""Select the routing threshold tau on the validation split alone.

For a given stage-1 model and the trained heavy teacher, tau is the smallest-cost
value whose validation accuracy stays within 1 point of always-heavy validation
accuracy. Test data is never touched here; this is what keeps the headline
equivalence claims free of selection leakage.

Usage:
    python scripts/select_tau.py --dataset gsd --seed 0 --stage1 distilled
"""
import argparse
import os

import numpy as np
import torch

import timm
from eval_all import (
    CKPT_DIR,
    conf_correct,
    load_dataset,
    select_tau,
)
from fvcore.nn import FlopCountAnalysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsd", "freiburg"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage1", choices=["cheap", "distilled"], default="distilled")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_items, _test, ncls = load_dataset(args.dataset)

    dummy = torch.randn(1, 3, 224, 224).to(device)
    g_cheap = FlopCountAnalysis(
        timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device),
        dummy,
    ).total() / 1e9
    g_heavy = FlopCountAnalysis(
        timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device),
        dummy,
    ).total() / 1e9

    stage1 = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
    stage1.load_state_dict(torch.load(
        f"{CKPT_DIR}/{args.dataset}_{args.stage1}_seed{args.seed}.pt", map_location=device))

    heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
    heavy.load_state_dict(torch.load(
        f"{CKPT_DIR}/{args.dataset}_heavy_seed{args.seed}.pt", map_location=device))

    tau = select_tau(stage1, heavy, val_items, g_cheap, g_heavy, device)

    # Report the selected operating point on val for context
    c_v, k_v = conf_correct(stage1, val_items, "none", 0, device)
    _, kh_v = conf_correct(heavy, val_items, "none", 0, device)
    keep = c_v >= tau
    acc = np.where(keep, k_v, kh_v).mean()
    routed = 1 - keep.mean()
    gmacs = g_cheap + routed * g_heavy

    print(f"{args.dataset} seed{args.seed} stage1={args.stage1}")
    print(f"  tau            = {tau:.4f}")
    print(f"  val accuracy   = {acc * 100:.2f}% "
          f"(always-heavy val {kh_v.mean() * 100:.2f}%)")
    print(f"  routed on val  = {routed * 100:.1f}%")
    print(f"  val GMACs      = {gmacs:.3f} (always-heavy {g_heavy:.3f})")


if __name__ == "__main__":
    main()
