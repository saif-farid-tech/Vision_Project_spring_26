"""splits.py — train/val/test loader for Caltech-101 using the Tip-Adapter
style split JSON (`split_zhou_Caltech101.json`).

The CoOp / Tip-Adapter split bakes the train / val / test partition into a
single JSON, so we don't need per-class count manifests like Food-101 did.
This module exposes one helper, `load_split`, which mirrors the old Food-101
API:

    train_ds, val_ds = splits.load_split(
        data_root="data",
        train_transform=train_tf,
        val_transform=val_tf,
    )

It also exposes `load_test`, since Caltech-101 has an explicit held-out test
list (we don't try to repurpose the val set as test).
"""

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import Dataset

from datasets.caltech101 import Caltech101
from datasets.utils import DatasetWrapper


# Default torchvision-style train/val/test input size used by all stages.
DEFAULT_INPUT_SIZE = 224


class _TipDataset(Dataset):
    """Thin adapter from a Tip-Adapter Datum list to a torchvision-style
    (image_tensor, label) Dataset.

    Mirrors `DatasetWrapper` semantics but exposes the standard
    `__getitem__` -> (tensor, int) interface that the existing training
    loops in this repo already speak.

    Also exposes `.classes` (list[str], sorted by label) so existing code
    that did `dataset.classes` keeps working.
    """

    def __init__(
        self,
        data_source,
        classnames,
        transform,
        input_size: int = DEFAULT_INPUT_SIZE,
        is_train: bool = False,
    ):
        self._wrapper = DatasetWrapper(
            data_source=data_source,
            input_size=input_size,
            transform=transform,
            is_train=is_train,
            return_img0=False,
        )
        self.classes = list(classnames)
        self._impaths = [item.impath for item in data_source]
        self._labels = [item.label for item in data_source]

    def __len__(self) -> int:
        return len(self._wrapper)

    def __getitem__(self, idx: int):
        return self._wrapper[idx]

    @property
    def image_paths(self):
        return self._impaths

    @property
    def labels(self):
        return self._labels


def _build(
    data_root,
    train_transform,
    val_transform,
    test_transform,
    input_size: int,
    num_shots: int,
) -> Tuple[_TipDataset, _TipDataset, _TipDataset, list]:
    """Construct the underlying Tip-Adapter Caltech101 instance and wrap
    each split. Returned tuple: (train_full, val, test, classnames).

    When `num_shots > 0` the *first* element returned is the few-shot subset
    (matching Tip-Adapter); when `num_shots <= 0` it is the full labeled
    training set.
    """
    ds = Caltech101(root=str(data_root), num_shots=num_shots)
    train_source = ds.train_x if num_shots > 0 else ds.train_full

    train_ds = _TipDataset(
        train_source, ds.classnames, train_transform,
        input_size=input_size, is_train=True,
    )
    val_ds = _TipDataset(
        ds.val, ds.classnames, val_transform,
        input_size=input_size, is_train=False,
    )
    test_ds = _TipDataset(
        ds.test, ds.classnames, test_transform,
        input_size=input_size, is_train=False,
    )
    return train_ds, val_ds, test_ds, ds.classnames


def load_split(
    data_root,
    train_transform=None,
    val_transform=None,
    input_size: int = DEFAULT_INPUT_SIZE,
    num_shots: int = -1,
    # `split_json` is accepted for backwards compat with the old Food-101 API,
    # but is unused: the Tip-Adapter split path is fixed at
    # <data_root>/caltech-101/split_zhou_Caltech101.json.
    split_json: Optional[str] = None,
) -> Tuple[_TipDataset, _TipDataset]:
    """Return (train_ds, val_ds) for Caltech-101 using the Tip-Adapter split."""
    train_ds, val_ds, _test_ds, _ = _build(
        data_root, train_transform, val_transform, val_transform,
        input_size=input_size, num_shots=num_shots,
    )
    return train_ds, val_ds


def load_test(
    data_root,
    transform=None,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> _TipDataset:
    """Return the held-out test split (matching evaluate_all.py's needs)."""
    _train_ds, _val_ds, test_ds, _ = _build(
        data_root, transform, transform, transform,
        input_size=input_size, num_shots=-1,
    )
    return test_ds


def load_all(
    data_root,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    input_size: int = DEFAULT_INPUT_SIZE,
    num_shots: int = -1,
) -> Tuple[_TipDataset, _TipDataset, _TipDataset, list]:
    """Return (train, val, test, classnames) — convenience for scripts that
    want every split in one call."""
    return _build(
        data_root, train_transform, val_transform,
        test_transform if test_transform is not None else val_transform,
        input_size=input_size, num_shots=num_shots,
    )


def ensure_prepared(data_root, seed: int = 1) -> Path:
    """Trigger Caltech-101 download / split generation if needed and return
    the dataset root. Safe to call from any script."""
    return Caltech101.auto_prepare(data_root, seed=seed)
