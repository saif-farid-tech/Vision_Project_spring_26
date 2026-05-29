"""prepare_ucf101.py
Download UCF101 (Soomro et al., 2012 — 101 human-action classes) in the
CoOp / Tip-Adapter "mid-frame still images" formulation, and make sure
the Tip-Adapter style split JSON exists.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/ucf101/UCF-101-midframes/<Action_Underscored>/<image>.jpg
    <root>/ucf101/split_zhou_UCF101.json

The script will:
  1. Try to fetch the mid-frame archive (`UCF-101-midframes.zip`) and the
     official CoOp split JSON via gdown from CoOp's shared Drive folder.
  2. If the mid-frame archive is already unpacked at the canonical path,
     skip the download step.
  3. If the split JSON cannot be downloaded, generate a deterministic
     per-class 50/20/30 train/val/test split locally from whatever is
     present under `UCF-101-midframes/`.
  4. If neither auto-download nor a pre-existing mid-frame tree is
     available, error out with manual-download instructions.

Usage:
    python prepare_ucf101.py [--root data] [--seed 1] [--force-generate-split]
"""

import argparse
from pathlib import Path

from datasets.ucf101 import UCF101


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare UCF101 (mid-frame form)")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/ucf101/",
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

    ds_root = UCF101.auto_prepare(
        args.root,
        seed=args.seed,
        force_generate_split=args.force_generate_split,
    )

    ds = UCF101(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
