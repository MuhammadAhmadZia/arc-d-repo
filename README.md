# ARC-D: Efficient Adaptive Cascade Inference for Automated Product Identification Without Barcodes


Reference implementation and experiment artifacts for the paper
**"Efficient Adaptive Cascade Inference for Automated Product Identification Without Barcodes"**
(invited extension of Ng et al., ICiMR 2025).

ARC-D is a two-stage classification cascade for barcode-free retail product recognition. A knowledge-distilled EfficientNet-B0 classifies every image, and only samples whose max-softmax confidence falls below a validation-selected threshold τ are routed to a ConvNeXt-Tiny second stage. On the Grocery Store Dataset ARC-D is statistically equivalent (TOST ±1 point) to always-heavy at 42–58% lower compute; on Freiburg Groceries it operates at 72–81% lower compute with 0.5–0.7 point clean-accuracy differences. Ablations rule out alternative routing signals and threshold policies; hardware measurements across two GPU generations and several CPU configurations show that operation counts overstate real savings by roughly 2× on GPUs and unpredictably on CPUs.

All experiments run on free public compute (Google Colab, Kaggle).

## Contents

```
scripts/                  # standalone Python scripts, one job per file
  train_all.py            # trains all models for one dataset × seed
  eval_all.py             # evaluates 5 systems × 4 conditions from checkpoints
  select_tau.py           # picks routing threshold on validation split
  hw_measure.py           # portable hardware measurement (T4, P100, CPU)
  pooled_tost.py          # per-image pooled equivalence test across seeds
  dual_confidence.py      # base-paper mechanism as an evaluated baseline
notebooks/                # end-to-end Colab / Kaggle notebooks
  final_runs_arcd.ipynb   # single source of truth for all paper numbers
  hw_measurement.ipynb    # portable hardware measurement, appends CSV
  extA_dual_confidence.ipynb
  pooled_tost_eval.ipynb
  figures/make_figures.py # regenerates Figures 2, 3, 4, 5, 6
results/                  # CSVs used to build every paper table
  final_results.csv       # dataset × seed × system × condition × accuracy / GMACs / τ
  dual_confidence_results.csv
  hw_results.csv          # runtime × device × metric × ratio × savings
checkpoints/              # best-val weights per (dataset, model, seed)
  README.md               # download links (weights not committed; too large for git)
.zenodo.json              # metadata for automatic DOI on tagged releases
```

## Quick start

Reproduce the paper's numbers from scratch. Fine on a Colab T4; total runtime is roughly 4–5 hours split across sessions and resumable.

```bash
git clone https://github.com/MuhammadAhmadZia/arc-d.git
cd arc-d && pip install -r requirements.txt

# 1. Train all models for both datasets and three seeds (resumable)
python scripts/train_all.py --dataset gsd      --seeds 0 1 2
python scripts/train_all.py --dataset freiburg --seeds 0 1 2

# 2. Reproduce every accuracy / GMACs number in Tables 2–4
python scripts/eval_all.py --dataset gsd      --seeds 0 1 2 --out results/final_results.csv
python scripts/eval_all.py --dataset freiburg --seeds 0 1 2 --out results/final_results.csv

# 3. Reproduce the dual-confidence baseline row (Ng et al.)
python scripts/dual_confidence.py --datasets gsd freiburg --seeds 0 1 2

# 4. Pooled equivalence test (Table 4)
python scripts/pooled_tost.py --datasets gsd freiburg

# 5. Hardware measurement — run once per target runtime, appends to CSV
python scripts/hw_measure.py --out results/hw_results.csv
```

For hardware measurement in particular, running the same script on Colab T4, Kaggle P100, and a Colab CPU runtime reproduces Table 5 in the paper.

## Datasets

Both datasets are natural in-store retail photographs. Neither is redistributed here.

- **Grocery Store Dataset** — Klasson et al., WACV 2019. 81 fine-grained classes, ~5k images. https://github.com/marcusklasson/GroceryStoreDataset
- **Freiburg Groceries Dataset** — Jund et al., 2016. 25 packaged-goods categories, ~5k images. http://aisdatasets.informatik.uni-freiburg.de

`scripts/train_all.py` downloads both automatically the first time it is run.

## Method summary

| Component | Setting |
|---|---|
| Stage-1 backbone | EfficientNet-B0 (~4.11 M params, 0.42 GMACs) |
| Stage-2 backbone | ConvNeXt-Tiny (~27.88 M params, 4.47 GMACs) |
| Initialization | ImageNet-pretrained (`timm`) |
| Optimizer / schedule | AdamW; cosine annealing over 20 epochs |
| Learning rate | 3e-4 (EfficientNet-B0); 1e-4 (ConvNeXt-Tiny) |
| Batch size / augmentation | 32; horizontal flip (p=0.5) |
| Distillation | KL on teacher logits, T=4, α=0.7, equal 20-epoch budget |
| Threshold τ | Smallest-cost value within 1 point of always-heavy val accuracy; validation only |
| Seeds | 0, 1, 2; data splits and corruption patterns fixed across seeds |

## Reproducibility notes

- **Free-tier only.** All results were produced on Google Colab (T4, CPU) and Kaggle (P100, CPU). No dedicated hardware is required.
- **Checkpoint discipline.** Per-epoch resume state and best-val weights are written after every epoch. Interrupted sessions cost at most one epoch.
- **Threshold leakage.** τ is selected on the validation split only; test data plays no role in threshold selection.
- **Hardware labels.** `hw_measure.py` records GPU model, CPU model, core count, and PyTorch thread count in every row. Within-device ratios are the trustworthy quantity; absolute energy readings are not comparable across devices.
- **Kaggle P100 note.** The Pascal architecture is not supported by the default PyTorch build on current Kaggle runtimes. `hw_measure.py` detects the P100 and installs a compatible build (torch 2.4.0 + cu118) automatically.
- **One-epoch resume caveat.** A resumed run restores model, optimizer, and scheduler state exactly, but not the data-shuffle RNG position, so batch order after resume differs slightly from an uninterrupted run.

## Citation

If this work is useful in your research, please cite the extended paper (details will be added upon publication) and the base conference paper:

```bibtex
@article{arcd2026,
  title   = {Efficient Adaptive Cascade Inference for Automated Product Identification Without Barcodes},
  year    = {2026},
  note    = {Extended version of Ng et al., ICiMR 2025. Full BibTeX to appear upon publication.}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Dataset licenses follow the original providers.

## DOI

Every tagged release of this repository is archived automatically on Zenodo, which mints a citable DOI.

## Acknowledgements

This work extends the ICiMR 2025 conference paper of Ng et al. Repository maintenance by Ahmad Zia ([www.ahmadzia.com](https://www.ahmadzia.com)). Compute was provided by Google Colab and Kaggle at their free tiers.

