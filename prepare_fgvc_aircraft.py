"""prepare_fgvc_aircraft.py
Download FGVC Aircraft (Maji et al., 2013) and re-layout it into the
Tip-Adapter style folder convention.

Final on-disk layout (mirroring Tip-Adapter):
    <root>/fgvc_aircraft/images/<id>.jpg
    <root>/fgvc_aircraft/variants.txt
    <root>/fgvc_aircraft/images_variant_train.txt
    <root>/fgvc_aircraft/images_variant_val.txt
    <root>/fgvc_aircraft/images_variant_test.txt

FGVC Aircraft ships its own canonical train / val / test split files, so
unlike the Caltech / Flowers / Cars / Pets variants there is **no CoOp
JSON split JSON** to download or generate — the loader reads
`images_variant_{train,val,test}.txt` directly.

Usage:
    python prepare_fgvc_aircraft.py [--root data]
"""

import argparse
from pathlib import Path

from datasets.fgvc import FGVCAircraft


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare FGVC Aircraft")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Parent directory; dataset is placed under <root>/fgvc_aircraft/",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)

    ds_root = FGVCAircraft.auto_prepare(args.root)

    ds = FGVCAircraft(root=str(args.root), num_shots=-1)
    print(f"\nDone. Dataset root: {ds_root}")
    print(f"  classes : {ds.num_classes}")
    print(f"  train   : {len(ds.train_full):,}")
    print(f"  val     : {len(ds.val):,}")
    print(f"  test    : {len(ds.test):,}")


if __name__ == "__main__":
    main()
