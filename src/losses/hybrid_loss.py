"""Hybrid BCE + IoU + SSIM loss, as popularized for saliency detection by
BASNet (Qin et al., 2019). Plain binary cross-entropy treats every pixel
independently and is dominated by the background class, since salient pixels
are a small minority of each EORSSD mask -- a model can rack up high pixel
"accuracy" while visually outputting a near-blank mask, because accuracy on a
heavily imbalanced pixel-classification task is a weak signal. IoU and SSIM
terms push the loss to actually care about the shape and structure of the
salient region, not just per-pixel correctness."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # 1,1,K,K


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM(pred, target), both single-channel in [0, 1]."""
    kernel = _gaussian_kernel(window_size, sigma=1.5, device=pred.device, dtype=pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)

    sigma_p = F.conv2d(pred * pred, kernel, padding=pad) - mu_p**2
    sigma_t = F.conv2d(target * target, kernel, padding=pad) - mu_t**2
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad) - mu_p * mu_t

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
        (mu_p**2 + mu_t**2 + c1) * (sigma_p + sigma_t + c2)
    )
    return 1 - ssim_map.mean()


def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return 1 - iou.mean()


class HybridLoss(nn.Module):
    def __init__(self, bce_weight=1.0, iou_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.ssim_weight = ssim_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        iou = iou_loss(probs, target)
        ssim = ssim_loss(probs, target)
        total = self.bce_weight * bce + self.iou_weight * iou + self.ssim_weight * ssim
        return total, {"bce": bce.item(), "iou": iou.item(), "ssim": ssim.item()}
