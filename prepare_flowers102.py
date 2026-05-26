"""prepare_flowers102.py
Download Oxford Flowers 102 and make sure the Tip-Adapter style split JSON
exists.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/oxford_flowers/jpg/image_XXXXX.jpg
    <root>/oxford_flowers/imagelabels.mat
    <root>/oxford_flowers/cat_to_name.json
    <root>/oxford_flowers/split_zhou_OxfordFlowers.json

The split JSON is the same format used by the CoOp / Tip-Adapter project.

The script will:
  1. Use `torchvision.datasets.Flowers102` to download the dataset if the
     `jpg/` directory or `imagelabels.mat` is missing.
  2. Try to fetch the official CoOp split + cat_to_name.json via gdown.
  3. Otherwise, write a hardcoded `cat_to_name.json` and generate a
     deterministic per-class 50/20/30 split from `imagelabels.mat`.

Usage:
    python prepare_flowers102.py [--root data] [--seed 1] [--force-generate-split]
"""

import argparse
from pathlib import Path

from datasets.oxford_flowers import OxfordFlowers


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare Oxford Flowers 102")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/oxford_flowers/",
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

    ds_root = OxfordFlowers.auto_prepare(
        args.root,
        seed=args.seed,
        force_generate_split=args.force_generate_split,
    )

    ds = OxfordFlowers(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
