# Results

Every file in this directory records results that appear in the paper, so a
reader can look up an exact number, curve, or matrix without rerunning the
experiments. Each file is reproducible by a script in `scripts/`; none is an
input to any code.

## `final_results.csv`

One row per (dataset, seed, system, condition). Produced by `eval_all.py`. This
is the source of truth for Tables 2, 3, and 4.

| Column | Meaning |
|---|---|
| dataset | `gsd` or `freiburg` |
| seed | training seed (0, 1, or 2) |
| system | `cheap-B0`, `distilled-B0`, `heavy-Tiny`, `ARC`, or `ARC-D` |
| condition | `clean`, `jpeg_s2`, `blur_s2`, or `occ_s1` |
| acc | test accuracy on this condition, in [0, 1] |
| gmacs | average GMACs per image for this system on this condition |
| routed | fraction of test images routed to stage 2 (ARC and ARC-D only) |
| tau | routing threshold, selected on validation data (ARC and ARC-D only) |
| tost_mean | mean paired accuracy difference vs. always-heavy (equivalence test) |
| tost_lo | lower bound of the 90% bootstrap confidence interval |
| tost_hi | upper bound of the 90% bootstrap confidence interval |

The `tost_*` columns back the equivalence tests in Table 4. Equivalence holds
when both bounds lie inside the one-percentage-point margin.

## `dual_confidence_results.csv`

The conference paper's dual-execution mechanism, re-implemented at ARC-D's
backbone scale as an evaluated baseline. One row per (dataset, seed, condition).
Produced by `dual_confidence.py`. Same schema as above minus the routing and
equivalence columns.

## `hw_results.csv` and the per-device files

`hw_results.csv` holds the combined hardware measurements; the per-device files
`colab-t4-hw_results.csv`, `kaggle-p100-hw_results.csv`, and
`colab-cpu-hw_results.csv` hold the individual runs behind them. Produced by
`hw_measure.py`, run once per target runtime. Together these are the source of
Table 5.

| Column | Meaning |
|---|---|
| env | detected platform: `colab`, `kaggle`, or `other` |
| device | GPU name (e.g. `Tesla T4`) or CPU model with core/thread count |
| metric | `GMACs`, `gpu_latency_ms`, `gpu_energy`, or `cpu_latency_ms` |
| cheap | measured stage-1 (EfficientNet-B0) value |
| heavy | measured stage-2 (ConvNeXt-Tiny) value |
| ratio | `cheap / heavy`; the cascade saves iff `ratio < 1 - f` |
| save_f0.3 / save_f0.4 / save_f0.43 / save_f0.6 | percent saving at that routed fraction |

The stage cost ratio of 0.089 in the paper is `cheap / heavy` on the measured
stage-1 cost of 0.40 GMACs (0.398 before rounding); the nominal architectural
figure quoted elsewhere is 0.42 GMACs.

## `severity_sweep.csv`

The complete five-level corruption sweep. The main tables report one
representative severity per family; this file gives the full degradation
curves. Produced by `scripts/severity_sweep.py`, run once per dataset.

| Column | Meaning |
|---|---|
| dataset | `gsd` or `freiburg` |
| seed | training seed (0, 1, or 2) |
| family | `blur`, `jpeg`, or `occlusion` |
| severity | corruption level, 1 to 5 |
| arcd_acc | ARC-D cascade accuracy at this severity, in [0, 1] |
| heavy_acc | always-heavy accuracy at this severity, in [0, 1] |
| routed | fraction of images routed to stage 2 at this severity |

## `error_analysis/`

Per-class accuracy and confusion matrices for ARC-D on both datasets, showing
which product classes are confused and the dominant error modes (near-identical
produce and same-brand flavor variants). Produced by `scripts/error_analysis.py`,
run once per dataset.

| File | Contents |
|---|---|
| `gsd_confusion_matrix.png`, `freiburg_confusion_matrix.png` | confusion matrices, log scale |
| `gsd_per_class_accuracy.csv`, `freiburg_per_class_accuracy.csv` | per-class accuracy with support count |
| `gsd_top_confusions.csv`, `freiburg_top_confusions.csv` | off-diagonal confusions, ranked by count |
| `gsd_worst_classes.png`, `freiburg_worst_classes.png` | the twenty lowest-accuracy classes |

## Comparing energy across devices

Do not compare absolute energy readings across different GPU or CPU rows. Only
the within-device `ratio` is used for any claim in the paper. Power is sampled
differently across environments, so absolute values are not directly comparable,
but the ratio of two forward passes measured on the same device under the same
protocol is trustworthy.
