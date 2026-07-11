"""Portable hardware measurement: run once per target runtime.

Auto-detects the platform (Colab, Kaggle, other) from environment variables,
labels every row with the exact device model, pins CPU threads to core count,
and appends to results/hw_results.csv. Handles Kaggle's Pascal-incompatible
PyTorch build by installing a cu118 build when a P100 is detected.

Measures four ratios of E_cheap / E_heavy:
    GMACs             (architectural, dataset-agnostic)
    gpu_latency_ms    (saturated batch, warm-up, synchronized)
    gpu_energy_kgco2e (NVML tracking via codecarbon)
    cpu_latency_ms    (single image, warm-up)

CPU energy is deliberately not reported because hosted environments expose no
reliable CPU power counter.

Usage (on the target runtime):
    python scripts/hw_measure.py --out results/hw_results.csv
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
import time


def detect_env():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "other"


def ensure_pascal_torch():
    """Kaggle P100 needs cu118 torch; default cu128 build has no sm_60 kernels."""
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    if "P100" in smi:
        print("P100 detected: installing torch 2.4.0 + cu118 (~3-4 min)")
        subprocess.run(
            [sys.executable, "-m", "pip", "-q", "install",
             "torch==2.4.0", "torchvision==0.19.0",
             "--index-url", "https://download.pytorch.org/whl/cu118"],
            check=True,
        )
    return smi


def gpu_latency(model, device, bs=64, warmup=10, iters=50):
    import torch
    m = model.to(device).eval()
    x = torch.randn(bs, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / (iters * bs) * 1000


def gpu_energy(model, device, bs=64, iters=200):
    import torch
    from codecarbon import EmissionsTracker
    m = model.to(device).eval()
    x = torch.randn(bs, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
    tracker.start()
    with torch.no_grad():
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
    kg = tracker.stop()
    return kg / (iters * bs)


def cpu_latency(model, warmup=2, iters=None):
    import torch
    m = model.to("cpu").eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        t = time.time(); m(x); per = time.time() - t
        it = iters or max(5, min(15, int(30 / max(per, 0.05))))  # cap at ~30s
        for _ in range(warmup):
            m(x)
        t0 = time.time()
        for _ in range(it):
            m(x)
    return (time.time() - t0) / it * 1000


def cpu_model_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hw_results.csv")
    args = ap.parse_args()

    env = detect_env()
    gpu_smi = ensure_pascal_torch() if env == "kaggle" else ""

    # Import torch AFTER any P100 fixup, so we load the installed build.
    import torch
    import timm
    from fvcore.nn import FlopCountAnalysis

    torch.set_num_threads(os.cpu_count())  # standardize CPU thread count

    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "none"
    cpu_m = cpu_model_name()
    cores = os.cpu_count()
    threads = torch.get_num_threads()

    print(f"env={env}  gpu={gpu_name}  cpu={cpu_m} ({cores}c/{threads}t)  torch={torch.__version__}")

    cheap = timm.create_model("efficientnet_b0", pretrained=False, num_classes=81).eval()
    heavy = timm.create_model("convnext_tiny",   pretrained=False, num_classes=81).eval()

    d1 = torch.randn(1, 3, 224, 224)
    g_cheap = FlopCountAnalysis(cheap, d1).total() / 1e9
    g_heavy = FlopCountAnalysis(heavy, d1).total() / 1e9

    rows = [("GMACs", g_cheap, g_heavy, "architectural")]
    if has_gpu:
        lc, lh = gpu_latency(cheap, "cuda"), gpu_latency(heavy, "cuda")
        rows.append(("gpu_latency_ms", lc, lh, gpu_name))
        try:
            ec, eh = gpu_energy(cheap, "cuda"), gpu_energy(heavy, "cuda")
            rows.append(("gpu_energy_kgco2e", ec, eh, gpu_name))
        except Exception as ex:
            print(f"gpu energy skipped: {ex}")
        cheap.to("cpu"); heavy.to("cpu"); torch.cuda.empty_cache()

    cc, ch = cpu_latency(cheap), cpu_latency(heavy)
    rows.append(("cpu_latency_ms", cc, ch, f"{cpu_m} ({cores}c/{threads}t)"))

    print(f"\n{'metric':20s} cheap        heavy       ratio")
    for name, a, b, _ in rows:
        print(f"{name:20s} {a:.4g}    {b:.4g}    {a / b:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["timestamp", "env", "device", "metric", "cheap", "heavy", "ratio",
              "save_f0.30", "save_f0.40", "save_f0.43", "save_f0.60"]
    new = not os.path.exists(args.out)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for name, a, b, dev in rows:
            r = a / b
            savings = {f"save_f{fr:.2f}": 100 * (1 - (r + fr)) for fr in (0.30, 0.40, 0.43, 0.60)}
            w.writerow(dict(timestamp=stamp, env=env, device=dev,
                            metric=name, cheap=a, heavy=b, ratio=r, **savings))
    print(f"\nappended {len(rows)} rows to {args.out}")
    print("cascade saves iff ratio < 1 - f")


if __name__ == "__main__":
    main()
