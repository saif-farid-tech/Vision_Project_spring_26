"""prepare_stanford_cars.py
Download Stanford Cars and make sure the Tip-Adapter style split JSON exists.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/stanford_cars/cars_train/<image>.jpg
    <root>/stanford_cars/cars_test/<image>.jpg
    <root>/stanford_cars/devkit/cars_meta.mat
    <root>/stanford_cars/devkit/cars_train_annos.mat
    <root>/stanford_cars/cars_test_annos_withlabels.mat
    <root>/stanford_cars/split_zhou_StanfordCars.json

The script will:
  1. Try to download via `torchvision.datasets.StanfordCars`.  NOTE: the
     original Stanford Cars host went offline in late 2022, so this often
     fails — in that case, manually download the dataset from a mirror
     (Kaggle / HuggingFace) and unpack it into `<root>/stanford_cars/`
     with the expected layout, then re-run this script.
  2. Try to fetch the official CoOp split via gdown.
  3. Otherwise, build a deterministic per-class 50/20/30 split using the
     dataset's `cars_meta.mat` + `cars_*_annos.mat` files.

Usage:
    python prepare_stanford_cars.py [--root data] [--seed 1] [--force-generate-split]
"""

import argparse
from pathlib import Path

from datasets.stanford_cars import StanfordCars


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare Stanford Cars")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/stanford_cars/",
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

    ds_root = StanfordCars.auto_prepare(
        args.root,
        seed=args.seed,
        force_generate_split=args.force_generate_split,
    )

    ds = StanfordCars(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
