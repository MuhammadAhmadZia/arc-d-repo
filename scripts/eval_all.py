"""Evaluate all 5 systems on the test split, under clean and 3 corruption conditions.

Reads trained checkpoints from checkpoints/, picks tau on the val split, then appends
one row per (dataset, seed, system, condition) to results/final_results.csv.

Usage:
    python scripts/eval_all.py --dataset gsd      --seeds 0 1 2 --out results/final_results.csv
    python scripts/eval_all.py --dataset freiburg --seeds 0 1 2 --out results/final_results.csv
"""
import argparse
import csv
import os
import random
import subprocess
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
from fvcore.nn import FlopCountAnalysis

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CKPT_DIR = "checkpoints"


def to_tensor_norm(img):
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]
    return t


def corrupt(img, kind, sev, idx=0):
    """Deterministic corruption: same image for every model given (kind, sev, idx)."""
    if kind == "none" or sev == 0:
        return img
    if kind == "blur":
        k = [3, 5, 7, 11, 15][sev - 1]
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "jpeg":
        q = [40, 30, 20, 12, 7][sev - 1]
        _, enc = cv2.imencode(".jpg",
                              cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    if kind == "occlusion":
        s = [0.1, 0.2, 0.3, 0.4, 0.5][sev - 1]
        side = int(224 * s)
        rng = random.Random(10_000 * sev + idx)
        x0, y0 = rng.randint(0, 224 - side), rng.randint(0, 224 - side)
        img = img.copy()
        img[y0:y0 + side, x0:x0 + side] = 127
        return img
    raise ValueError(kind)


class GroceryDS(Dataset):
    def __init__(self, items, kind="none", sev=0):
        self.items, self.kind, self.sev = items, kind, sev

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
        return to_tensor_norm(corrupt(img, self.kind, self.sev, idx=i)), label


def load_dataset(name):
    if name == "gsd":
        if not os.path.exists("GroceryStoreDataset"):
            subprocess.run(
                ["git", "clone", "-q",
                 "https://github.com/marcusklasson/GroceryStoreDataset.git"],
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
        return rd("val.txt"), rd("test.txt"), 81

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
        seen = set(); out = []
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
    rng = random.Random(0)
    by_class = {}
    for it in trainval:
        by_class.setdefault(it[1], []).append(it)
    val = []
    for lst in by_class.values():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, int(0.1 * len(lst)))
        val += lst[:k]
    return val, test, 25


@torch.no_grad()
def conf_correct(model, items, kind, sev, device, bs=64):
    """Return (max_confidence, correct) arrays for the whole set."""
    model.eval()
    dl = DataLoader(GroceryDS(items, kind, sev), batch_size=bs, num_workers=2, pin_memory=True)
    C, K = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(device)), 1)
        c, pred = p.max(1)
        C.append(c.cpu().numpy())
        K.append((pred.cpu() == y).numpy())
    return np.concatenate(C), np.concatenate(K)


def select_tau(stage1, heavy, val_items, g_cheap, g_heavy, device):
    """Pick the cheapest tau whose val accuracy stays within 1 point of always-heavy val."""
    c_v, k_v = conf_correct(stage1, val_items, "none", 0, device)
    _, kh_v = conf_correct(heavy, val_items, "none", 0, device)
    target = kh_v.mean() - 0.01
    best = None
    for t in np.concatenate([[0.0], np.linspace(0.5, 0.999, 60)]):
        keep = c_v >= t
        acc = np.where(keep, k_v, kh_v).mean()
        g = g_cheap + (1 - keep.mean()) * g_heavy
        if acc >= target and (best is None or g < best[1]):
            best = (t, g, acc)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsd", "freiburg"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/final_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_items, test_items, ncls = load_dataset(args.dataset)
    print(f"{args.dataset}: val={len(val_items)}  test={len(test_items)}  classes={ncls}")

    # GMACs are architectural, not dataset-dependent
    dummy = torch.randn(1, 3, 224, 224).to(device)
    g_cheap = FlopCountAnalysis(
        timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device),
        dummy,
    ).total() / 1e9
    g_heavy = FlopCountAnalysis(
        timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device),
        dummy,
    ).total() / 1e9
    print(f"GMACs: cheap/distilled={g_cheap:.3f}  heavy={g_heavy:.3f}")

    conditions = [("clean", "none", 0), ("jpeg_s2", "jpeg", 2),
                  ("blur_s2", "blur", 2), ("occ_s1", "occlusion", 1)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["dataset", "seed", "system", "condition", "acc", "gmacs", "routed", "tau"]
    new = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()

        for seed in args.seeds:
            print(f"===== {args.dataset} seed {seed} =====")
            cheap = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
            cheap.load_state_dict(torch.load(
                f"{CKPT_DIR}/{args.dataset}_cheap_seed{seed}.pt", map_location=device))
            distilled = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
            distilled.load_state_dict(torch.load(
                f"{CKPT_DIR}/{args.dataset}_distilled_seed{seed}.pt", map_location=device))
            heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
            heavy.load_state_dict(torch.load(
                f"{CKPT_DIR}/{args.dataset}_heavy_seed{seed}.pt", map_location=device))

            tau_arc = select_tau(cheap, heavy, val_items, g_cheap, g_heavy, device)
            tau_arcd = select_tau(distilled, heavy, val_items, g_cheap, g_heavy, device)
            print(f"  tau ARC={tau_arc:.3f}  tau ARC-D={tau_arcd:.3f}")

            for tag, kind, sev in conditions:
                cc, kc = conf_correct(cheap, test_items, kind, sev, device)
                cd, kd = conf_correct(distilled, test_items, kind, sev, device)
                _, kh = conf_correct(heavy, test_items, kind, sev, device)

                keep_arc = cc >= tau_arc
                keep_arcd = cd >= tau_arcd
                acc_arc = np.where(keep_arc, kc, kh).mean()
                acc_arcd = np.where(keep_arcd, kd, kh).mean()
                r_arc = 1 - keep_arc.mean()
                r_arcd = 1 - keep_arcd.mean()

                for system, acc, g, routed, tau in [
                    ("cheap-B0", kc.mean(), g_cheap, "", ""),
                    ("distilled-B0", kd.mean(), g_cheap, "", ""),
                    ("heavy-Tiny", kh.mean(), g_heavy, "", ""),
                    ("ARC", acc_arc, g_cheap + r_arc * g_heavy, r_arc, tau_arc),
                    ("ARC-D", acc_arcd, g_cheap + r_arcd * g_heavy, r_arcd, tau_arcd),
                ]:
                    w.writerow(dict(dataset=args.dataset, seed=seed, system=system,
                                    condition=tag, acc=acc, gmacs=g, routed=routed, tau=tau))
                print(f"  {tag:8s} ARC-D {acc_arcd * 100:5.1f}% @ "
                      f"{g_cheap + r_arcd * g_heavy:4.2f} GMACs  "
                      f"(heavy {kh.mean() * 100:5.1f}%)")

            del cheap, distilled, heavy
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
