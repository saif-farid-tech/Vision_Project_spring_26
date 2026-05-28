"""Tip-Adapter style dataset utilities (vendored).

Source: https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets

Vendored locally so the FGVC Aircraft experiment can use the same Datum /
DatasetBase / DatasetWrapper abstractions. Unlike the other dataset variants
in this project, FGVC Aircraft ships its own canonical train/val/test split
files, so there is no CoOp JSON split involved.
"""

from .utils import (
    Datum,
    DatasetBase,
    DatasetWrapper,
    build_data_loader,
    read_image,
    read_json,
    write_json,
    listdir_nohidden,
)
from .fgvc import FGVCAircraft

__all__ = [
    "Datum",
    "DatasetBase",
    "DatasetWrapper",
    "build_data_loader",
    "read_image",
    "read_json",
    "write_json",
    "listdir_nohidden",
    "FGVCAircraft",
]
