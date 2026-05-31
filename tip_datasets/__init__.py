"""tip_datasets — Tip-Adapter's data preprocessing + split, ported into this repo.

This package mirrors the relevant parts of
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/ so that the
``_Food101`` branch reuses Tip-Adapter's exact data pipeline:

  * the **Zhou train/val/test split** (``split_zhou_Food101.json``), and
  * the **CLIP preprocessing** (BICUBIC resize + ToTensor + CLIP normalization)

instead of the previous torchvision 90/10 count-manifest split and ImageNet
normalization.
"""

from .utils import (
    Datum,
    DatasetBase,
    DatasetWrapper,
    build_data_loader,
    read_image,
    read_json,
    write_json,
)
from .oxford_pets import OxfordPets
from .food101 import Food101

__all__ = [
    "Datum",
    "DatasetBase",
    "DatasetWrapper",
    "build_data_loader",
    "read_image",
    "read_json",
    "write_json",
    "OxfordPets",
    "Food101",
]
