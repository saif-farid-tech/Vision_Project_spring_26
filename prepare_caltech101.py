"""prepare_caltech101.py
Download Caltech-101 and make sure the Tip-Adapter style split JSON exists.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/caltech-101/101_ObjectCategories/<class>/<image>.jpg
    <root>/caltech-101/split_zhou_Caltech101.json

The split JSON is the same format used by the CoOp / Tip-Adapter project:
    {
      "train": [[relative_image_path, label, classname], ...],
      "val":   [...],
      "test":  [...],
    }

The script will:
  1. Use `torchvision.datasets.Caltech101` to download the dataset if the
     image directory is missing.
  2. Try to fetch the official CoOp split via gdown.
  3. If gdown is unavailable or the download fails, generate a deterministic
     per-class 50/20/30 split locally.

Usage:
    python prepare_caltech101.py [--root data] [--seed 1] [--force-generate-split]
"""

import argparse
from pathlib import Path

from datasets.caltech101 import Caltech101


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare Caltech-101")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/caltech-101/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed used for the locally-generated split (only if the "
             "official CoOp split cannot be downloaded).",
    )
    parser.add_argument(
        "--force-generate-split",
        action="store_true",
        help="Always generate the split locally; skip the gdown attempt.",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)

    ds_root = Caltech101.auto_prepare(
        args.root,
        seed=args.seed,
        force_generate_split=args.force_generate_split,
    )

    # Sanity check by actually loading the split.
    ds = Caltech101(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
