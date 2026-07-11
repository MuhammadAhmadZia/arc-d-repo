"""Train cheap-B0, heavy-Tiny, and distilled-B0 for one dataset and one or more seeds.

Resumable: writes best-val weights (dataset_model_seedN.pt) and a full per-epoch
resume state (dataset_model_seedN_resume.pt) after every epoch. On rerun, a completed
model (best.pt without resume file) is skipped; an interrupted one resumes at the
next epoch. τ selection and evaluation are handled by select_tau.py and eval_all.py.

Usage:
    python scripts/train_all.py --dataset gsd      --seeds 0 1 2
    python scripts/train_all.py --dataset freiburg --seeds 0
"""
import argparse
import os
import random
import subprocess
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import cv2  # noqa: F401 (used by corruption code in eval)
import timm

EPOCHS = 20
BATCH = 32
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def to_tensor_norm(img):
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]
    return t


class GroceryDS(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
        if self.train and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
        return to_tensor_norm(img), label


def load_gsd():
    if not os.path.exists("GroceryStoreDataset"):
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/marcusklasson/GroceryStoreDataset.git"],
            check=True,
        )
    root = "GroceryStoreDataset/dataset"

    def rd(txt):
        out = []
        with open(os.path.join(root, txt)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                out.append((os.path.join(root, parts[0]), int(parts[1])))
        return out

    return rd("train.txt"), rd("val.txt"), rd("test.txt"), 81


def load_freiburg():
    if not os.path.exists("images"):
        url = ("http://aisdatasets.informatik.uni-freiburg.de/"
               "freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz")
        print("downloading Freiburg (~1GB)...")
        urllib.request.urlretrieve(url, "fg.tar.gz")
        subprocess.run(["tar", "-xf", "fg.tar.gz"], check=True)
    raw = "https://raw.githubusercontent.com/PhilJd/freiburg_groceries_dataset/master/splits"
    for f in ["train0.txt", "test0.txt"]:
        if not os.path.exists(f):
            urllib.request.urlretrieve(f"{raw}/{f}", f)

    def rd(txt):
        seen = set()
        out = []
        with open(txt) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rel, lab = line.rsplit(" ", 1)
                if rel in seen:
                    continue
                seen.add(rel)
                out.append((os.path.join("images", rel), int(lab)))
        return out

    trainval, test = rd("train0.txt"), rd("test0.txt")
    # fixed-seed stratified 10% val carve, so it is identical across training seeds
    rng = random.Random(0)
    by_class = {}
    for it in trainval:
        by_class.setdefault(it[1], []).append(it)
    train, val = [], []
    for lst in by_class.values():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, int(0.1 * len(lst)))
        val += lst[:k]
        train += lst[k:]
    return train, val, test, 25


@torch.no_grad()
def val_acc(model, items, device):
    model.eval()
    dl = DataLoader(GroceryDS(items), batch_size=64, num_workers=2, pin_memory=True)
    ok = n = 0
    for x, y in dl:
        ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        n += y.size(0)
    return ok / n


def train_one(name, seed, dataset, lr, num_classes, train_items, val_items, device, teacher=None):
    ckpt = os.path.join(CKPT_DIR, f"{dataset}_{name}_seed{seed}.pt")
    res = os.path.join(CKPT_DIR, f"{dataset}_{name}_seed{seed}_resume.pt")

    # completed earlier: best.pt present, no resume file
    if os.path.exists(ckpt) and not os.path.exists(res):
        print(f"  {name} seed{seed}: already complete, skipping")
        return

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = timm.create_model(
        "efficientnet_b0" if name != "heavy" else "convnext_tiny",
        pretrained=True, num_classes=num_classes,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    start_ep, best = 0, -1.0

    if os.path.exists(res):
        state = torch.load(res, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_ep, best = state["epoch"] + 1, state["best"]
        print(f"  {name} seed{seed}: RESUMING from epoch {start_ep + 1} (best val {best * 100:.1f}%)")

    dl = DataLoader(GroceryDS(train_items, train=True), batch_size=BATCH,
                    shuffle=True, num_workers=2, pin_memory=True)

    for ep in range(start_ep, EPOCHS):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if teacher is None:
                loss = F.cross_entropy(logits, y)
            else:
                with torch.no_grad():
                    tl = teacher(x)
                soft = F.kl_div(F.log_softmax(logits / 4.0, 1),
                                F.softmax(tl / 4.0, 1),
                                reduction="batchmean") * 16.0
                loss = 0.7 * soft + 0.3 * F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        va = val_acc(model, val_items, device)
        star = ""
        if va > best:
            best = va
            torch.save(model.state_dict(), ckpt)
            star = " *"
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(),
             "sched": sched.state_dict(), "epoch": ep, "best": best},
            res,
        )
        print(f"  {name} s{seed} ep{ep + 1}/{EPOCHS} loss={run / n:.4f} "
              f"val={va * 100:.1f}%{star} ({time.time() - t0:.0f}s)")

    if os.path.exists(res):
        os.remove(res)
    print(f"  {name} seed{seed}: best val {best * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsd", "freiburg"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  dataset: {args.dataset}  seeds: {args.seeds}")

    train_items, val_items, _test, ncls = (
        load_gsd() if args.dataset == "gsd" else load_freiburg()
    )
    print(f"train={len(train_items)}  val={len(val_items)}  classes={ncls}")

    for seed in args.seeds:
        print(f"===== {args.dataset} seed {seed} =====")
        # Teacher (heavy) first, so the distilled student can load it.
        train_one("heavy", seed, args.dataset, 1e-4, ncls, train_items, val_items, device)
        train_one("cheap", seed, args.dataset, 3e-4, ncls, train_items, val_items, device)
        # Load the just-trained teacher for distillation.
        teacher = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
        teacher.load_state_dict(
            torch.load(os.path.join(CKPT_DIR, f"{args.dataset}_heavy_seed{seed}.pt"),
                       map_location=device)
        )
        teacher.eval()
        train_one("distilled", seed, args.dataset, 3e-4, ncls,
                  train_items, val_items, device, teacher=teacher)


if __name__ == "__main__":
    main()
