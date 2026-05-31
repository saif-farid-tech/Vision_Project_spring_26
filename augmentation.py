"""augmentation.py  (Tip-Adapter / _Food101 variant)

Data preprocessing for this branch is taken directly from Tip-Adapter's
``datasets/utils.py`` (the ``DatasetWrapper.to_tensor`` pipeline):

    BICUBIC Resize  ->  ToTensor  ->  CLIP normalization

No RandAugment, no RandomErasing, no Mixup/CutMix — train and validation use the
**same** deterministic transform, exactly as Tip-Adapter does. This replaces the
previous SHViT recipe (ImageNet normalization + RandAugment + RandomErasing +
Mixup) so the experiment matches Tip-Adapter's preprocessing.

``build_mixup_fn`` is kept (returns ``None``) only so existing imports in the
training scripts keep working; the scripts fall back to plain cross-entropy when
no mixup function is returned.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# CLIP normalization statistics (Tip-Adapter / OpenAI CLIP), NOT ImageNet's.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _tip_preprocess(img_size=224):
    """The Tip-Adapter DatasetWrapper preprocessing: BICUBIC resize to a square
    ``img_size``, ToTensor, then CLIP normalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_train_transform(img_size=224):
    """Training preprocessing — identical to Tip-Adapter (no augmentation)."""
    return _tip_preprocess(img_size)


def build_val_transform(img_size=224):
    """Validation / test preprocessing — identical to the train transform."""
    return _tip_preprocess(img_size)


def build_mixup_fn(num_classes=101):
    """Tip-Adapter uses no Mixup/CutMix; return ``None`` so callers fall back to
    plain cross-entropy on hard labels."""
    return None
