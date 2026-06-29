#!/usr/bin/env python
"""Evaluate one or all trained models on the EORSSD test split.

Single model:
    python scripts/evaluate.py --model unet --checkpoint checkpoints/unet_best.pth

All models with checkpoints present in --checkpoint-dir:
    python scripts/evaluate.py --all --checkpoint-dir checkpoints
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.eorssd import EORSSDDataset
from src.models.registry import MODEL_NAMES, get_model
from src.utils.metrics import SODMetricAccumulator
from src.utils.visualize import denormalize, save_comparison_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="EORSSD-Dataset/EORSSD")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model", choices=MODEL_NAMES)
    p.add_argument("--checkpoint")
    p.add_argument("--all", action="store_true")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--num-qualitative", type=int, default=8)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, device, save_dir: Path = None, num_qualitative: int = 0):
    model.eval()
    acc = SODMetricAccumulator()
    saved = 0
    for images, masks in loader:
        images_dev = images.to(device)
        logits = model(images_dev)
        probs = torch.sigmoid(logits).cpu().numpy()
        gts = masks.numpy()

        for i in range(images.size(0)):
            pred_map = probs[i, 0]
            gt_map = gts[i, 0]
            acc.update(pred_map, gt_map)

            if save_dir is not None and saved < num_qualitative:
                img_rgb = denormalize(images[i].numpy())
                save_comparison_grid(save_dir / f"sample_{saved:03d}.png", img_rgb, gt_map, pred_map)
                saved += 1

    return acc.compute()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = EORSSDDataset(args.data_root, "test", args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = [(name, Path(args.checkpoint_dir) / f"{name}_best.pth") for name in MODEL_NAMES]
        targets = [(n, c) for n, c in targets if c.exists()]
        if not targets:
            raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    else:
        if not args.model or not args.checkpoint:
            raise ValueError("Provide --model and --checkpoint, or use --all")
        targets = [(args.model, Path(args.checkpoint))]

    results = {}
    for name, ckpt_path in targets:
        print(f"Evaluating {name} ({ckpt_path})...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = get_model(name, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])

        sample_dir = output_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_model(model, test_loader, device, sample_dir, args.num_qualitative)
        results[name] = metrics
        print(f"  {metrics}")

    csv_path = output_dir / "comparison.csv"
    fieldnames = ["model", "MAE", "max-F", "mean-F", "S-measure", "E-measure"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in results.items():
            row = {"model": name, **m}
            writer.writerow(row)
    print(f"\nWrote comparison table to {csv_path}")


if __name__ == "__main__":
    main()
