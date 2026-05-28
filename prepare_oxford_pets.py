"""prepare_oxford_pets.py
Download the Oxford-IIIT Pet dataset and make sure the Tip-Adapter style
split JSON exists.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/oxford_pets/images/<Breed>_<id>.jpg
    <root>/oxford_pets/annotations/{trainval.txt, test.txt, ...}
    <root>/oxford_pets/split_zhou_OxfordPets.json

The split JSON is the same format used by the CoOp / Tip-Adapter project.

The script will:
  1. Use `torchvision.datasets.OxfordIIITPet` to download the dataset (both
     the `trainval` and `test` splits trigger the same tar) if the images
     or annotation files are missing.
  2. Try to fetch the official CoOp split via gdown.
  3. Otherwise, generate a deterministic per-class 50/20/30 train/val/test
     split locally from the dataset's annotation .txt files.

Usage:
    python prepare_oxford_pets.py [--root data] [--seed 1] [--force-generate-split]
"""

import argparse
from pathlib import Path

from datasets.oxford_pets import OxfordPets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare Oxford-IIIT Pets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/oxford_pets/",
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

    ds_root = OxfordPets.auto_prepare(
        args.root,
        seed=args.seed,
        force_generate_split=args.force_generate_split,
    )

    ds = OxfordPets(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
