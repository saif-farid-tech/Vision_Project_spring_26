"""Tip-Adapter style dataset utilities (vendored).

Source: https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets

Vendored locally so the Caltech-101 experiment can use the same Datum /
DatasetBase / DatasetWrapper abstractions and the CoOp-style split JSON.
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
from .caltech101 import Caltech101
from .oxford_pets import OxfordPets

__all__ = [
    "Datum",
    "DatasetBase",
    "DatasetWrapper",
    "build_data_loader",
    "read_image",
    "read_json",
    "write_json",
    "listdir_nohidden",
    "Caltech101",
    "OxfordPets",
]
