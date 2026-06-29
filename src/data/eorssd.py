"""EORSSD dataset loading.

Folder layout expected under `root`:
    train-images/*.jpg   train-labels/*.png
    test-images/*.jpg    test-labels/*.png

Images and labels are paired by filename stem (e.g. 0001.jpg <-> 0001.png).
Labels are single-channel binary saliency masks (0 = background, 255 = salient).
"""
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_eval_transforms

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_paired_files(image_dir: Path, label_dir: Path):
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    pairs = []
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.png"
        if not label_path.exists():
            continue
        pairs.append((img_path, label_path))
    if not pairs:
        raise FileNotFoundError(
            f"No matching image/label pairs found between {image_dir} and {label_dir}"
        )
    return pairs


class EORSSDDataset(Dataset):
    """Returns (image, mask) tensors. image: float32 CHW normalized.
    mask: float32 1HW in [0, 1]."""

    def __init__(self, root, split: str, img_size: int = 256, augment: bool = False, pairs=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        if pairs is not None:
            self.pairs = pairs
        else:
            image_dir = self.root / f"{split}-images"
            label_dir = self.root / f"{split}-labels"
            self.pairs = _list_paired_files(image_dir, label_dir)

        self.transform = (
            get_train_transforms(img_size) if augment else get_eval_transforms(img_size)
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {label_path}")
        mask = (mask > 127).astype(np.float32)  # binarize -> {0, 1}

        transformed = self.transform(image=image, mask=mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].unsqueeze(0).float()
        return image_t, mask_t

    @property
    def filenames(self):
        return [p[0].name for p in self.pairs]


def make_train_val_split(root, val_fraction: float, seed: int = 42):
    """Splits the EORSSD train set into train/val pair lists (val held out for
    model selection, separate from the official test split)."""
    root = Path(root)
    pairs = _list_paired_files(root / "train-images", root / "train-labels")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_fraction))
    val_idx = set(order[:n_val].tolist())
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]
    return train_pairs, val_pairs


def build_datasets(root, img_size: int, val_fraction: float, seed: int = 42):
    train_pairs, val_pairs = make_train_val_split(root, val_fraction, seed)
    train_ds = EORSSDDataset(root, "train", img_size, augment=True, pairs=train_pairs)
    val_ds = EORSSDDataset(root, "train", img_size, augment=False, pairs=val_pairs)
    test_ds = EORSSDDataset(root, "test", img_size, augment=False)
    return train_ds, val_ds, test_ds
