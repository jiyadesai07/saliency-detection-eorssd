#!/usr/bin/env python
"""Run a trained model on a single raw image and save the heatmap overlay:
input an RGB image, get back a saliency heatmap.

    python scripts/predict.py --model attention_unet \
        --checkpoint checkpoints/attention_unet_best.pth \
        --image some_satellite_image.jpg --output heatmap.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import get_eval_transforms
from src.models.registry import MODEL_NAMES, get_model
from src.utils.visualize import make_heatmap_overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="heatmap_overlay.png")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = get_eval_transforms(args.img_size)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(args.model, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap, overlay = make_heatmap_overlay(image_rgb, prob_map)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), overlay)
    cv2.imwrite(str(out_path.with_stem(out_path.stem + "_heatmap_only")), heatmap)
    print(f"Saved overlay to {out_path} and raw heatmap alongside it.")


if __name__ == "__main__":
    main()
