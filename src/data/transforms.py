"""Albumentations pipelines for EORSSD. Aerial imagery has no canonical
"up" direction, so we lean on flips/rotations rather than perspective warps
that would distort object shapes."""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
