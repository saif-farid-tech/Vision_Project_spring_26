"""splits.py  (Tip-Adapter / _Food101 variant)

Data split for this branch is Tip-Adapter's **Zhou split** for Food-101,
loaded from ``<data_root>/food-101/split_zhou_Food101.json`` via
``OxfordPets.read_split`` (see the ``tip_datasets`` package). This replaces the
previous deterministic 90/10 count-manifest split keyed off
``train_val_split_seed*.json``.

The directory layout is the one ``torchvision.datasets.Food101`` already
produces — ``<data_root>/food-101/images/<class>/<img>.jpg`` — plus the
``split_zhou_Food101.json`` file dropped into ``<data_root>/food-101/``.

Datasets are returned as ``DatasetWrapper`` instances (Tip-Adapter's loader),
each yielding ``(image_tensor, int_label)`` so they slot straight into the
existing training / evaluation loops.

Usage:
    import splits
    train_ds, val_ds = splits.load_split(
        data_root="data",
        train_transform=train_tf,
        val_transform=val_tf,
    )
    test_ds = splits.load_test("data", transform=val_tf)
    class_names = splits.get_class_names("data")   # label-index order
"""

from augmentation import build_train_transform, build_val_transform
from tip_datasets import DatasetWrapper, Food101

# Default split file name (lives under <data_root>/food-101/). Kept here so the
# notebooks can reference the canonical name when downloading it.
SPLIT_FILE_NAME = "split_zhou_Food101.json"


def build_food101(data_root, num_shots=-1):
    """Return the Tip-Adapter ``Food101`` dataset object (Zhou split).

    ``num_shots=-1`` keeps the full training split (no few-shot subsampling).
    The object exposes ``.train_x``, ``.val``, ``.test`` (lists of ``Datum``),
    plus ``.classnames`` / ``.lab2cname`` / ``.num_classes``.
    """
    return Food101(str(data_root), num_shots=num_shots)


def load_split(data_root, split_json=None, train_transform=None,
               val_transform=None, num_shots=-1, input_size=224):
    """Return ``(train_ds, val_ds)`` for the Zhou split.

    ``split_json`` is accepted (and ignored) only for backward compatibility
    with the previous count-manifest API; the split now always comes from
    ``<data_root>/food-101/split_zhou_Food101.json``.
    """
    base = build_food101(data_root, num_shots=num_shots)

    if train_transform is None:
        train_transform = build_train_transform(input_size)
    if val_transform is None:
        val_transform = build_val_transform(input_size)

    train_ds = DatasetWrapper(
        base.train_x, input_size=input_size, transform=train_transform, is_train=True,
    )
    val_ds = DatasetWrapper(
        base.val, input_size=input_size, transform=val_transform, is_train=False,
    )
    return train_ds, val_ds


def load_test(data_root, transform=None, num_shots=-1, input_size=224):
    """Return the Zhou-split **test** set as a ``DatasetWrapper``."""
    base = build_food101(data_root, num_shots=num_shots)
    if transform is None:
        transform = build_val_transform(input_size)
    return DatasetWrapper(
        base.test, input_size=input_size, transform=transform, is_train=False,
    )


def get_class_names(data_root, num_shots=-1):
    """Class names in label-index order (matches the trained model's logits)."""
    return list(build_food101(data_root, num_shots=num_shots).classnames)


def get_test_datums(data_root, num_shots=-1):
    """The ordered list of test ``Datum`` objects (impath / label / classname).

    Useful for error analysis / demos that need the original image paths in the
    same order the test ``DataLoader`` iterates them.
    """
    return build_food101(data_root, num_shots=num_shots).test
