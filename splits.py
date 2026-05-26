"""splits.py — train/val/test loader for Stanford Cars using the Tip-Adapter
style split JSON (`split_zhou_StanfordCars.json`).

The CoOp / Tip-Adapter split bakes the train / val / test partition into a
single JSON, so we don't need per-class count manifests. This module exposes
the same API as the previous dataset variants:

    train_ds, val_ds = splits.load_split(
        data_root="data",
        train_transform=train_tf,
        val_transform=val_tf,
    )

`load_test` returns the held-out test split, and `ensure_prepared` triggers
the download / split generation if needed.
"""

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import Dataset

from datasets.stanford_cars import StanfordCars
from datasets.utils import DatasetWrapper


DEFAULT_INPUT_SIZE = 224


class _TipDataset(Dataset):
    """Thin adapter from a Tip-Adapter Datum list to a torchvision-style
    (image_tensor, label) Dataset.

    Exposes `.classes` (list[str], sorted by label) so existing code
    that does `dataset.classes` keeps working.
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
    """Construct the underlying Tip-Adapter StanfordCars instance and wrap
    each split. Returned tuple: (train_full_or_fewshot, val, test, classnames).
    """
    ds = StanfordCars(root=str(data_root), num_shots=num_shots)
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
    split_json: Optional[str] = None,  # ignored — kept for API compat
) -> Tuple[_TipDataset, _TipDataset]:
    """Return (train_ds, val_ds) for Stanford Cars using the Tip-Adapter split."""
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
    """Trigger Stanford Cars download / split generation if needed and return
    the dataset root. Safe to call from any script."""
    return StanfordCars.auto_prepare(data_root, seed=seed)
