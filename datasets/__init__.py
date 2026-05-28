"""Tip-Adapter style dataset utilities (vendored).

Source: https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets

Vendored locally so the DTD experiment can use the same Datum / DatasetBase
/ DatasetWrapper abstractions and the CoOp-style split JSON.
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
from .dtd import DescribableTextures
from .oxford_pets import OxfordPets

# Alias the Tip-Adapter class under a friendlier name for code that imports
# `DTD` from this package.
DTD = DescribableTextures

__all__ = [
    "Datum",
    "DatasetBase",
    "DatasetWrapper",
    "build_data_loader",
    "read_image",
    "read_json",
    "write_json",
    "listdir_nohidden",
    "DescribableTextures",
    "DTD",
    "OxfordPets",
]
