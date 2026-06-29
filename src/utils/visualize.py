"""Heatmap visualization -- this is the actual end-user deliverable: feed in
a raw RGB image, get back a saliency heatmap overlay."""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    """image_chw: float array (3, H, W), normalized. Returns uint8 HWC RGB."""
    img = image_chw.transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def make_heatmap_overlay(image_rgb_u8: np.ndarray, prob_map: np.ndarray, alpha: float = 0.45):
    """image_rgb_u8: (H, W, 3) uint8. prob_map: (H, W) float in [0, 1].
    Returns (heatmap_bgr, overlay_bgr) both uint8, ready for cv2.imwrite."""
    heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    return heatmap, overlay


def save_comparison_grid(path, image_rgb_u8, gt_mask, prob_map):
    """Builds a 4-panel [image | ground truth | predicted heatmap | overlay]
    figure and writes it to `path`."""
    h, w = image_rgb_u8.shape[:2]
    gt_vis = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap, overlay = make_heatmap_overlay(image_rgb_u8, prob_map)
    image_bgr = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2BGR)
    panel = np.concatenate([image_bgr, gt_vis, heatmap, overlay], axis=1)
    cv2.imwrite(str(path), panel)
