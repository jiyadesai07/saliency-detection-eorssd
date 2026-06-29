#!/usr/bin/env python
"""Quick CPU sanity check: every model produces the right output shape, the
loss is finite and differentiable, and metrics compute without error. Not a
substitute for real training -- just verifies nothing is structurally broken
before handing this off to a GPU notebook."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.hybrid_loss import HybridLoss
from src.models.registry import MODEL_NAMES, get_model
from src.utils.metrics import SODMetricAccumulator

torch.manual_seed(0)


def check_model(name, pretrained):
    print(f"--- {name} (pretrained={pretrained}) ---")
    model = get_model(name, pretrained=pretrained)
    model.train()
    x = torch.randn(2, 3, 256, 256)
    y = (torch.rand(2, 1, 256, 256) > 0.7).float()

    logits = model(x)
    assert logits.shape == (2, 1, 256, 256), f"bad shape {logits.shape}"

    criterion = HybridLoss()
    loss, parts = criterion(logits, y)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert n_grad > 0, "no gradients flowed"
    print(f"  output shape OK, loss={loss.item():.4f} parts={parts}, params_with_grad={n_grad}")


def check_metrics():
    print("--- metrics ---")
    acc = SODMetricAccumulator(num_thresholds=16)
    rng = np.random.default_rng(0)
    for _ in range(3):
        pred = rng.random((64, 64)).astype(np.float32)
        gt = (rng.random((64, 64)) > 0.8).astype(np.float32)
        acc.update(pred, gt)
    result = acc.compute()
    print(f"  {result}")
    assert all(np.isfinite(v) for v in result.values())


if __name__ == "__main__":
    for name in MODEL_NAMES:
        # Use pretrained=False to avoid downloading ImageNet weights during the
        # smoke test; real training will set pretrained=True per config.
        check_model(name, pretrained=False)
    check_metrics()
    print("\nAll smoke tests passed.")
