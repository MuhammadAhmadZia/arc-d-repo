"""
ARC-D paper figures. Figure 1 (pipeline) is authored in PowerPoint by the authors,
so it is not generated here. This script produces:
  figure2a_effnet.png, figure2b_convnext.png  (architecture, two separate images)
  figure3_corruptions.png                     (3x3 matrix on the Arla yoghurt sample)
  figure4_pareto.png                          (two charts stacked in one column)
  figure5_routing.png
  figure6_savings.png
All at 300 dpi. Numbers are the final published values.
"""
import os, random, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import cv2

DPI = 300
SAMPLE_URL = ("https://raw.githubusercontent.com/marcusklasson/GroceryStoreDataset/"
              "refs/heads/master/dataset/iconic-images-and-descriptions/Packages/"
              "Yoghurt/Arla-Natural-Yoghurt/Arla-Natural-Yoghurt_Iconic.jpg")

def load_sample():
    fn = "arla_yoghurt.jpg"
    if not os.path.exists(fn):
        urllib.request.urlretrieve(SAMPLE_URL, fn)
    from PIL import Image
    return np.array(Image.open(fn).convert("RGB").resize((224, 224)))

# -------------------------------------------------- Figure 2a / 2b (architecture)
def _draw_tower(blocks, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 6.5))
    ax.axis("off"); ax.set_title(title, fontsize=10)
    n = len(blocks)
    for i, (name, ch, sp) in enumerate(blocks):
        y = 1 - (i + 1) / (n + 1)
        shaded = ("MBConv" in name) or ("block" in name)
        ax.add_patch(FancyBboxPatch((0.06, y - 0.033), 0.88, 0.062,
                     boxstyle="round,pad=0.008",
                     fc="#cfe3f7" if shaded else "#eeeeee", ec="k", lw=0.8))
        ax.text(0.5, y, f"{name}   |   ch {ch}   |   out {sp}",
                ha="center", va="center", fontsize=8.5)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - 0.045), xytext=(0.5, y - 0.033),
                        arrowprops=dict(arrowstyle="-|>", lw=1))
    ax.text(0.5, 0.985, "input 224x224x3", ha="center", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(fname, dpi=DPI, bbox_inches="tight"); plt.close(fig)

def figure2_architecture():
    b0 = [("stem conv 3x3, s2", "32", "112x112"), ("MBConv1 3x3", "16", "112x112"),
          ("MBConv6 3x3 x2", "24", "56x56"), ("MBConv6 5x5 x2", "40", "28x28"),
          ("MBConv6 3x3 x3", "80", "14x14"), ("MBConv6 5x5 x3", "112", "14x14"),
          ("MBConv6 5x5 x4", "192", "7x7"), ("MBConv6 3x3 x1", "320", "7x7"),
          ("conv 1x1 + pool", "1280", "1x1"), ("FC head", "N classes", "-")]
    cnx = [("stem conv 4x4, s4", "96", "56x56"), ("stage 1: 3 blocks", "96", "56x56"),
           ("downsample", "192", "28x28"), ("stage 2: 3 blocks", "192", "28x28"),
           ("downsample", "384", "14x14"), ("stage 3: 9 blocks", "384", "14x14"),
           ("downsample", "768", "7x7"), ("stage 4: 3 blocks", "768", "7x7"),
           ("global pool + LN", "768", "1x1"), ("FC head", "N classes", "-")]
    _draw_tower(b0,  "EfficientNet-B0 stage 1 (about 4.11 M params, 0.42 GMACs)",
                "figure2a_effnet.png")
    _draw_tower(cnx, "ConvNeXt-Tiny stage 2 (about 27.88 M params, 4.47 GMACs)",
                "figure2b_convnext.png")

# -------------------------------------------------- Figure 3 (3x3 corruption matrix)
def corrupt(im, kind, sev):
    if kind == "blur":
        k = [3, 5, 7, 11, 15][sev-1]; return cv2.GaussianBlur(im, (k, k), 0)
    if kind == "jpeg":
        q = [40, 30, 20, 12, 7][sev-1]
        _, e = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(e, 1), cv2.COLOR_BGR2RGB)
    if kind == "occlusion":
        s = [0.1, 0.2, 0.3, 0.4, 0.5][sev-1]; side = int(224*s)
        r = random.Random(42); x0 = r.randint(0, 224-side); y0 = r.randint(0, 224-side)
        im = im.copy(); im[y0:y0+side, x0:x0+side] = 127; return im
    return im

def figure3_corruptions():
    img = load_sample()
    families = [("Gaussian blur", "blur"), ("JPEG", "jpeg"), ("Occlusion", "occlusion")]
    sevs = [1, 3, 5]          # adjust here if other severities are preferred
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8))
    for r, (fname, kind) in enumerate(families):
        for c, sev in enumerate(sevs):
            ax = axes[r, c]
            ax.imshow(corrupt(img, kind, sev)); ax.axis("off")
            if r == 0:
                ax.set_title(f"severity {sev}", fontsize=9)
        axes[r, 0].set_ylabel(fname, fontsize=9)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig("figure3_corruptions.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

# -------------------------------------------------- Figure 4 (Pareto, column layout)
def figure4_pareto():
    gsd = {"EfficientNet-B0": (0.42, 80.1), "Distilled B0": (0.42, 86.7),
           "ConvNeXt-Tiny": (4.47, 92.3), "ARC": (2.38, 91.1), "ARC-D": (1.88, 92.0),
           "Dual-confidence": (4.89, 91.4)}
    fbg = {"EfficientNet-B0": (0.42, 86.4), "Distilled B0": (0.42, 89.9),
           "ConvNeXt-Tiny": (4.47, 93.9), "ARC": (1.73, 93.5), "ARC-D": (1.11, 93.3),
           "Dual-confidence": (4.89, 93.8)}
    marks = {"EfficientNet-B0": ("o", "tab:gray"), "Distilled B0": ("s", "tab:blue"),
             "ConvNeXt-Tiny": ("^", "tab:red"), "ARC": ("D", "tab:orange"),
             "ARC-D": ("*", "tab:green"), "Dual-confidence": ("v", "tab:purple")}
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    for ax, data, tag in zip(axes, (gsd, fbg),
                             ("(a) Grocery Store Dataset", "(b) Freiburg Groceries")):
        for name, (g, a) in data.items():
            m, c = marks[name]
            ax.scatter(g, a, marker=m, s=170 if m == "*" else 80, color=c,
                       label=name, zorder=3, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("average GMACs per image")
        ax.set_ylabel("clean accuracy (%)")
        ax.set_title(tag, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("figure4_pareto.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

# -------------------------------------------------- Figure 5 (routing fractions)
def figure5_routing():
    conds = ["clean", "JPEG s2", "blur s2", "occ s1"]
    gsd_arcd = [32.7, 48.7, 49.2, 32.7]
    fbg_arcd = [15.5, 26.9, 27.1, 17.6]
    x = np.arange(4); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w/2, gsd_arcd, w, label="GSD", color="tab:blue")
    ax.bar(x + w/2, fbg_arcd, w, label="Freiburg", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("ARC-D routed fraction (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig("figure5_routing.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

# -------------------------------------------------- Figure 6 (predicted vs measured)
def figure6_savings():
    labels = ["GMACs\n(predicted)", "T4\nlatency", "T4\nenergy", "P100\nlatency",
              "P100\nenergy", "CPU A\n(2c)", "CPU B\n(4c)", "CPU C\n(2c)"]
    vals = [48, 27, 28, 28, 24, 21, 14, -3]
    colors = ["tab:gray"] + ["tab:blue"]*4 + ["tab:orange"]*3
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (1.2 if v >= 0 else -3),
                f"{v:+d}%", ha="center", fontsize=8)
    ax.set_ylabel("cascade saving vs always-heavy (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig("figure6_savings.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    figure2_architecture(); figure3_corruptions()
    figure4_pareto(); figure5_routing(); figure6_savings()
    print("wrote figures 2a, 2b, 3, 4, 5, 6 (300 dpi); Figure 1 comes from PowerPoint")
