"""Per-class accuracy and confusion matrices for ARC-D (Reviewer 1.12).

Self-contained: regenerates results/error_analysis/ (per-class accuracy CSVs,
ranked confusion CSVs, confusion-matrix figures, worst-class bar charts) for
both datasets, from trained checkpoints. Runs the ARC-D cascade (distilled first
stage + routed heavy stage) so the error modes are those of the deployed system.

Usage:
    python scripts/error_analysis.py --dataset gsd      --seed 0 --tau 0.914
    python scripts/error_analysis.py --dataset freiburg --seed 0 --tau 0.90

Provide --tau (the validation-selected routing threshold reported in the paper),
or place results/final_results.csv alongside and it will be read automatically.
"""
import argparse, csv, os, subprocess, urllib.request
import numpy as np, torch, timm, matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CKPT_DIR = "checkpoints"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def norm(img):
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]
    return t


class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        return norm(np.array(Image.open(p).convert("RGB").resize((224, 224)))), l


def load_gsd():
    if not os.path.exists("GroceryStoreDataset"):
        subprocess.run(["git", "clone", "-q",
                        "https://github.com/marcusklasson/GroceryStoreDataset.git"], check=True)
    root = "GroceryStoreDataset/dataset"
    def rd(t):
        out = []
        for line in open(f"{root}/{t}"):
            line = line.strip()
            if line:
                p = [x.strip() for x in line.split(",")]
                out.append((f"{root}/{p[0]}", int(p[1])))
        return out
    names = {}
    with open(f"{root}/classes.csv") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                try:
                    names[int(parts[1])] = parts[0]
                except ValueError:
                    pass
    return rd("test.txt"), 81, names


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
                seen.add(rel)
                out.append((f"images/{rel}", int(lab)))
    names = {i: n for i, n in enumerate(
        sorted(set(os.path.basename(os.path.dirname(p)) for p, _ in out)))}
    return out, 25, names


@torch.no_grad()
def infer(model, items, device):
    model.eval()
    dl = DataLoader(DS(items), batch_size=64, num_workers=2)
    C, P, Y = [], [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(device)), 1)
        c, pred = p.max(1)
        C.append(c.cpu().numpy()); P.append(pred.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(C), np.concatenate(P), np.concatenate(Y)


def read_tau(results_csv, dataset):
    if not os.path.exists(results_csv):
        return None
    for row in csv.DictReader(open(results_csv)):
        if (row.get("dataset") == dataset and row.get("system") == "ARC-D"
                and row.get("condition") == "clean" and row.get("tau")):
            return float(row["tau"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsd", "freiburg"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--results", default="results/final_results.csv")
    ap.add_argument("--out", default="results/error_analysis")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    test, ncls, names = load_gsd() if args.dataset == "gsd" else load_freiburg()
    tau = args.tau if args.tau is not None else read_tau(args.results, args.dataset)
    if tau is None:
        raise SystemExit("No tau: pass --tau or provide results/final_results.csv")
    print(f"{args.dataset}: {len(test)} test images, {ncls} classes, tau={tau:.3f}")

    distilled = timm.create_model("efficientnet_b0", pretrained=False, num_classes=ncls).to(device)
    distilled.load_state_dict(torch.load(
        f"{CKPT_DIR}/{args.dataset}_distilled_seed{args.seed}.pt", map_location=device))
    heavy = timm.create_model("convnext_tiny", pretrained=False, num_classes=ncls).to(device)
    heavy.load_state_dict(torch.load(
        f"{CKPT_DIR}/{args.dataset}_heavy_seed{args.seed}.pt", map_location=device))

    c_d, p_d, y = infer(distilled, test, device)
    _, p_h, _ = infer(heavy, test, device)
    routed = c_d < tau
    pred = np.where(routed, p_h, p_d)
    print(f"  ARC-D accuracy {(pred == y).mean() * 100:.2f}%, routed {routed.mean() * 100:.1f}%")

    cm = np.zeros((ncls, ncls), dtype=int)
    for tr, pr in zip(y, pred):
        cm[tr, pr] += 1
    pc = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(ncls), where=cm.sum(1) > 0)
    ds = args.dataset

    with open(f"{args.out}/{ds}_per_class_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class_id", "class_name", "support", "accuracy"])
        for i in range(ncls):
            w.writerow([i, names.get(i, f"class_{i}"), int(cm[i].sum()), f"{pc[i]:.4f}"])
    conf = sorted(((cm[i, j], names.get(i, str(i)), names.get(j, str(j)))
                   for i in range(ncls) for j in range(ncls) if i != j and cm[i, j] > 0), reverse=True)
    with open(f"{args.out}/{ds}_top_confusions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["count", "true_class", "predicted_class"])
        for n, ti, tj in conf:
            w.writerow([n, ti, tj])

    fig, ax = plt.subplots(figsize=(8, 7) if ncls > 30 else (6, 5))
    im = ax.imshow(np.log1p(cm), cmap="viridis", aspect="auto")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{ds.upper()} confusion matrix (ARC-D, log scale)")
    fig.colorbar(im, label="log(1 + count)")
    plt.tight_layout(); plt.savefig(f"{args.out}/{ds}_confusion_matrix.png", dpi=200); plt.close()

    order = np.argsort(pc)[:20]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names.get(i, str(i)) for i in order], pc[order] * 100)
    ax.set_xlabel("Accuracy (%)"); ax.set_title(f"{ds.upper()} lowest-accuracy classes (ARC-D)")
    plt.tight_layout(); plt.savefig(f"{args.out}/{ds}_worst_classes.png", dpi=200); plt.close()
    print(f"  wrote 4 files to {args.out}/ for {ds}")


if __name__ == "__main__":
    main()
