"""splits.py – deterministic 90/10 train/val split using Ahmed's JSON manifests.

The JSON manifests (train_val_split_seed*.json) contain per-class counts, not
image paths.  This helper converts them into actual torch Subset indices by
shuffling each class's indices with the manifest's seed.

Food101(split='test') is never loaded here — it is reserved for the Stage 3
final evaluation only.

Usage:
    import splits
    train_ds, val_ds = splits.load_split(
        data_root="data",
        split_json="train_val_split_seed42.json",
        train_transform=train_tf,
        val_transform=val_tf,
    )
"""

import json
import random

import torchvision
from torch.utils.data import Subset


def load_split(data_root, split_json, train_transform=None, val_transform=None):
    """
    Returns (train_subset, val_subset) carved from Food101(split='train').

    Each class contributes exactly manifest["val"][class_name] samples to the
    val subset and the remainder to the train subset, selected by a
    deterministic shuffle keyed to manifest["seed"].

    Two separate Food101 instances are created so each subset can carry its
    own transform without interfering with the other.
    """
    with open(split_json) as f:
        manifest = json.load(f)
    seed = manifest["seed"]

    # Probe instance: no transform, just used to enumerate labels.
    probe = torchvision.datasets.Food101(
        root=str(data_root), split="train", download=True
    )
    idx_to_class = probe.classes  # sorted list; integer label == list position

    # Group dataset indices by class name.
    class_indices: dict = {c: [] for c in idx_to_class}
    for i, label in enumerate(probe._labels):
        class_indices[idx_to_class[label]].append(i)

    # Deterministic per-class shuffle, then split.
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls_name, indices in class_indices.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = manifest["val"][cls_name]
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    # Two instances so each subset has its own transform.
    train_ds = torchvision.datasets.Food101(
        root=str(data_root), split="train", transform=train_transform, download=False
    )
    val_ds = torchvision.datasets.Food101(
        root=str(data_root), split="train", transform=val_transform, download=False
    )
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx)
