"""augmentation.py — training / validation transforms and Mixup helper.

Normalization follows the Tip-Adapter convention (CLIP statistics), which is
also what the vendored datasets.utils.DatasetWrapper uses by default. This
keeps train / val / test pixel statistics consistent across every stage of
the experiment.
"""

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.mixup import Mixup


# CLIP / Tip-Adapter normalization (also used by datasets.utils.DatasetWrapper).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Oxford Flowers 102 has 102 classes (matching the dataset name).
NUM_CLASSES = 102


def build_train_transform(img_size: int = 224):
    """Training augmentations (RandAugment + RandomErasing). Uses bicubic
    interpolation for the resize, matching Tip-Adapter."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_val_transform(img_size: int = 224):
    """Validation / test transforms (resize + center crop)."""
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_mixup_fn(num_classes: int = NUM_CLASSES):
    """Mixup + CutMix + label smoothing (matches the SHViT recipe)."""
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=num_classes,
    )
