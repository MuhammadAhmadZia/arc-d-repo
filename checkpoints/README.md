# Checkpoints

Model weights are not committed to git because each ConvNeXt-Tiny checkpoint is
around 110 MB and the full set is over 1 GB. Two options for obtaining them.

## Option 1: reproduce from scratch (recommended)

Running `python scripts/train_all.py --dataset gsd --seeds 0 1 2` and the
matching Freiburg command from the repository root produces every checkpoint
this repo expects. Total training time is roughly 4-5 hours on a Colab T4 GPU
and is fully resumable.

The trained files land here as:

```
checkpoints/
  gsd_cheap_seed0.pt        gsd_cheap_seed1.pt        gsd_cheap_seed2.pt
  gsd_distilled_seed0.pt    gsd_distilled_seed1.pt    gsd_distilled_seed2.pt
  gsd_heavy_seed0.pt        gsd_heavy_seed1.pt        gsd_heavy_seed2.pt
  freiburg_cheap_seed0.pt   freiburg_cheap_seed1.pt   freiburg_cheap_seed2.pt
  freiburg_distilled_seed0.pt   freiburg_distilled_seed1.pt   freiburg_distilled_seed2.pt
  freiburg_heavy_seed0.pt   freiburg_heavy_seed1.pt   freiburg_heavy_seed2.pt
```

## Option 2: download pretrained weights

A release with the exact weights used in the paper will be attached to the
first tagged release of this repository. See the Releases page.

## File format

Each `.pt` file is a plain `state_dict` saved via `torch.save(model.state_dict(), path)`.
Load with the matching `timm` model:

```python
import timm, torch
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=81)
model.load_state_dict(torch.load("checkpoints/gsd_distilled_seed0.pt", map_location="cpu"))
model.eval()
```

The distilled and cheap files share the EfficientNet-B0 architecture; only the
weights differ (distilled = KD-trained from ConvNeXt-Tiny; cheap = plain
fine-tune from ImageNet).
