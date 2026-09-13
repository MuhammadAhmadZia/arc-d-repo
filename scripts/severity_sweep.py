"""Full five-level corruption severity sweep (manuscript: robustness / Limitations).

Self-contained. The main tables report one representative severity per corruption
family; the manuscript states the complete five-level sweep is in the repository.
This script produces it: for each dataset, seed, family, and severity 1-5, it
records ARC-D accuracy, always-heavy accuracy, and the routed fraction, so the
full degradation curves can be plotted.

Usage:
    python scripts/severity_sweep.py --dataset gsd      --seeds 0 1 2 --tau 0.914
    python scripts/severity_sweep.py --dataset freiburg --seeds 0 1 2 --tau 0.90
"""
import argparse, csv, os, subprocess, urllib.request, random
import numpy as np, torch, timm, cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CKPT_DIR = "checkpoints"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def norm(img):
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]
    return t


def corrupt(img, kind, sev, idx=0):
    if kind == "none" or sev == 0:
        return img
    if kind == "blur":
        k = [3, 5, 7, 11, 15][sev - 1]
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "jpeg":
        q = [40, 30, 20, 12, 7][sev - 1]
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    if kind == "occlusion":
        s = [0.1, 0.2, 0.3, 0.4, 0.5][sev - 1]
        side = int(224 * s)
        rng = random.Random(10_000 * sev + idx)
        x0, y0 = rng.randint(0, 224 - side), rng.randint(0, 224 - side)
        img = img.copy(); img[y0:y0 + side, x0:x0 + side] = 127
        return img
    raise ValueError(kind)


class DS(Dataset):
    def __init__(self, items, kind, sev): self.items, self.kind, self.sev = items, kind, sev
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        img = np.array(Image.open(p).convert("RGB").resize((224, 224)))
        return norm(corrupt(img, self.kind, self.sev, idx=i)), l


def load_gsd():
    if not os.path.exists("GroceryStoreDataset"):
        subprocess.run(["git", "clone", "-q",
                        "https://github.com/marcusklasson/GroceryStoreDataset.git"], check=True)
    root = "GroceryStoreDataset/dataset"
    out = []
    for line in open(f"{root}/test.txt"):
        line = line.strip()
        if line:
            p = [x.strip() for x in line.split(",")]
            out.append((f"{root}/{p[0]}", int(p[1])))
    return out, 81


def load_freiburg():
    if not os.path.exists("images"):
        url = ("http://aisdatasets.informatik.uni-freiburg.de/"
               "freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz")
        urllib.request.urlretrieve(url, "fg.tar.gz")
        subprocess.run(["tar", "-xf", "fg.tar.gz"], check=True)
    raw = "https://raw.githubusercontent.com/PhilJd/freiburg_groceries_dataset/master/splits"
    if not os.path.exists("test0.txt"):
        urllib.request.urlretrieve(f"{raw}/test0.txt", "test0.txt")
    seen, out = set(), []
    for line in open("test0.txt"):
        line = line.strip()
        if line:
            rel, lab = line.rsplit(" ", 1)
            if rel not in seen:
                seen.add(rel); out.append((f"images/{rel}", int(lab)))
    return out, 25


@torch.no_grad()
def conf_correct(model, items, kind, sev, device):
    model.eval()
    dl = DataLoader(DS(items, kind, sev), batch_size=64, num_workers=2)
    C, K = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(device)), 1)
        c, pred = p.max(1)
        C.append(c.cpu().numpy()); K.append((pred.cpu() == y).numpy())
    return np.concatenate(C), np.concatenate(K)


def read_tau(results_csv, dataset, seed):
    if not os.path.exists(results_csv):
        return None
    for row in csv.DictReader(open(results_csv)):
        if (row.get("dataset") == dataset and row.get("system") == "ARC-D"
                and row.get("condition") == "clean" and str(row.get("seed")) == str(seed)
                and row.get("tau")):
            return float(row["tau"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsd", "freiburg"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--results", default="results/final_results.csv")
    ap.add_argument("--out", default="results/severity_sweep.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test, ncls = load_gsd() if args.dataset == "gsd" else load_freiburg()
    families = ["blur", "jpeg", "occlusion"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "family", "severity",
                                          "arcd_acc", "heavy_acc", "routed"])
        if new:
            w.writeheader()
        for seed in args.seeds:
            tau = args.tau if args.tau is not None else (read_tau(args.results, args.dataset, seed) or 0.9)
            distilled = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
            distilled.load_state_dict(torch.load(
                f"{CKPT_DIR}/{args.dataset}_distilled_seed{seed}.pt", map_location=device))
            heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
            heavy.load_state_dict(torch.load(
                f"{CKPT_DIR}/{args.dataset}_heavy_seed{seed}.pt", map_location=device))
            for fam in families:
                for sev in range(1, 6):
                    c_d, k_d = conf_correct(distilled, test, fam, sev, device)
                    _, k_h = conf_correct(heavy, test, fam, sev, device)
                    routed = c_d < tau
                    arcd = np.where(routed, k_h, k_d).mean()
                    w.writerow(dict(dataset=args.dataset, seed=seed, family=fam, severity=sev,
                                    arcd_acc=f"{arcd:.4f}", heavy_acc=f"{k_h.mean():.4f}",
                                    routed=f"{routed.mean():.4f}"))
                    print(f"  {args.dataset} s{seed} {fam} sev{sev}: ARC-D {arcd*100:.1f}% "
                          f"heavy {k_h.mean()*100:.1f}% routed {routed.mean()*100:.0f}%")
            del distilled, heavy
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"\nsweep written to {args.out}")


if __name__ == "__main__":
    main()
