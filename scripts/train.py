#!/usr/bin/env python
"""Train a single saliency model on EORSSD.

Example:
    python scripts/train.py --config configs/default.yaml --model attention_unet
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.eorssd import build_datasets
from src.losses.hybrid_loss import HybridLoss
from src.models.registry import MODEL_NAMES, get_model
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", choices=MODEL_NAMES, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["train"]["model"] = args.model
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.img_size:
        cfg["data"]["img_size"] = args.img_size
    if args.no_pretrained:
        cfg["train"]["pretrained"] = False
    if args.output_dir:
        cfg["output"]["checkpoint_dir"] = args.output_dir
    return cfg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss, _ = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * images.size(0)
        total_mae += torch.abs(probs - masks).mean().item() * images.size(0)
        n += images.size(0)
    return total_loss / n, total_mae / n


def main():
    args = parse_args()
    cfg = load_config(args)

    set_seed(cfg["train"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, _ = build_datasets(
        cfg["data"]["root"], cfg["data"]["img_size"], cfg["data"]["val_fraction"], cfg["train"]["seed"]
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=(device == "cuda"),
    )

    model_name = cfg["train"]["model"]
    model = get_model(model_name, pretrained=cfg["train"]["pretrained"]).to(device)
    criterion = HybridLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    use_amp = cfg["train"]["amp"] and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}_log.csv"

    best_val_loss = float("inf")
    patience_left = cfg["train"]["early_stop_patience"]

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mae", "lr", "seconds"])

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        model.train()
        running_loss, n = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss, _ = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
        scheduler.step()
        train_loss = running_loss / n

        val_loss, val_mae = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{model_name}] epoch {epoch:03d}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f} "
            f"lr={lr_now:.2e} ({elapsed:.1f}s)"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_mae, lr_now, elapsed])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg["train"]["early_stop_patience"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": cfg},
                ckpt_dir / f"{model_name}_best.pth",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement for "
                      f"{cfg['train']['early_stop_patience']} epochs)")
                break

    print(f"Best val_loss={best_val_loss:.4f}. Checkpoint: {ckpt_dir / f'{model_name}_best.pth'}")


if __name__ == "__main__":
    main()
