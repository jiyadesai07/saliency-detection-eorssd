"""Saliency-detection evaluation metrics.

Pixel-wise "accuracy" is a weak signal on EORSSD: salient pixels are a small
minority of each mask, so a model that predicts mostly background still
scores high despite being qualitatively useless. The metrics below are the
standard protocol used in optical-remote-sensing saliency-detection
research (see MathLee/ORSI-SOD_Summary for a survey of the field): MAE,
max/mean F-measure, S-measure, and E-measure.
"""
import numpy as np


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred - gt).mean())


def _object_score(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    x = values.mean()
    sigma = values.std()
    return float(2 * x / (x**2 + 1 + sigma + 1e-8))


def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    fg_score = _object_score(pred[gt == 1])
    bg_score = _object_score((1 - pred)[gt == 0])
    u = gt.mean()
    return u * fg_score + (1 - u) * bg_score


def _ssim_region(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    n = pred.size
    x, y = pred.mean(), gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + 1e-8)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + 1e-8)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-8)
    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        return float(alpha / (beta + 1e-8))
    return 1.0 if beta == 0 else 0.0


def _centroid(gt: np.ndarray):
    if gt.sum() == 0:
        h, w = gt.shape
        return h // 2, w // 2
    ys, xs = np.where(gt > 0)
    return int(round(ys.mean())), int(round(xs.mean()))


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    cy, cx = _centroid(gt)
    cy = min(max(cy, 1), h - 1)
    cx = min(max(cx, 1), w - 1)

    quadrants = [
        (gt[:cy, :cx], pred[:cy, :cx]),
        (gt[:cy, cx:], pred[:cy, cx:]),
        (gt[cy:, :cx], pred[cy:, :cx]),
        (gt[cy:, cx:], pred[cy:, cx:]),
    ]
    total = h * w
    score = 0.0
    for g_r, p_r in quadrants:
        weight = g_r.size / total
        score += weight * _ssim_region(p_r, g_r)
    return score


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    y = gt.mean()
    if y == 0:
        return float(1 - pred.mean())
    if y == 1:
        return float(pred.mean())
    so = _s_object(pred, gt)
    sr = _s_region(pred, gt)
    return float(alpha * so + (1 - alpha) * sr)


def e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Adaptive (mean) E-measure: binarize pred at 2x its own mean, then
    score alignment against the ground truth."""
    gt_bin = (gt > 0.5).astype(np.float64)
    if gt_bin.sum() == 0:
        return float((1 - pred).mean())
    if gt_bin.sum() == gt_bin.size:
        return float(pred.mean())

    threshold = min(2 * pred.mean(), 1.0)
    pred_bin = (pred >= threshold).astype(np.float64)
    phi_fm = pred_bin - pred_bin.mean()
    phi_gt = gt_bin - gt_bin.mean()
    align = (2 * phi_fm * phi_gt) / (phi_fm**2 + phi_gt**2 + 1e-8)
    enhanced = ((align + 1) ** 2) / 4
    return float(enhanced.mean())


class SODMetricAccumulator:
    """Accumulates dataset-level MAE / S-measure / E-measure / F-measure
    (precision-recall averaged per threshold across the whole test set, as is
    standard for SOD benchmarking, rather than per-image)."""

    def __init__(self, num_thresholds: int = 256, beta2: float = 0.3):
        self.thresholds = np.linspace(0, 1, num_thresholds)
        self.beta2 = beta2
        self.precision_sum = np.zeros(num_thresholds)
        self.recall_sum = np.zeros(num_thresholds)
        self.mae_sum = 0.0
        self.s_sum = 0.0
        self.e_sum = 0.0
        self.n = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """pred, gt: 2D float arrays in [0, 1], identical shape."""
        self.mae_sum += mae(pred, gt)
        self.s_sum += s_measure(pred, gt)
        self.e_sum += e_measure(pred, gt)

        gt_bin = gt > 0.5
        gt_sum = gt_bin.sum()
        for i, t in enumerate(self.thresholds):
            pred_bin = pred >= t
            tp = np.logical_and(pred_bin, gt_bin).sum()
            fp = pred_bin.sum() - tp
            self.precision_sum[i] += tp / (tp + fp + 1e-8)
            self.recall_sum[i] += tp / (gt_sum + 1e-8)
        self.n += 1

    def compute(self) -> dict:
        if self.n == 0:
            raise RuntimeError("No samples accumulated")
        precision = self.precision_sum / self.n
        recall = self.recall_sum / self.n
        f_beta = ((1 + self.beta2) * precision * recall) / (
            self.beta2 * precision + recall + 1e-8
        )
        return {
            "MAE": self.mae_sum / self.n,
            "S-measure": self.s_sum / self.n,
            "E-measure": self.e_sum / self.n,
            "max-F": float(f_beta.max()),
            "mean-F": float(f_beta.mean()),
        }
