"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/fgvc.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.FGVCAircraft` download, and
  * re-shapes the unpack into the Tip-Adapter layout
    `<root>/fgvc_aircraft/{images/, variants.txt,
    images_variant_train.txt, images_variant_val.txt,
    images_variant_test.txt}`.

Unlike the Caltech / Flowers / Cars / Pets variants, FGVC Aircraft ships
its own canonical train / val / test split files, so there is no CoOp
split JSON involved here — the loader reads `images_variant_{train,val,
test}.txt` directly.
"""

import os
import shutil
from pathlib import Path

from .utils import Datum, DatasetBase


template = ["a photo of a {}, a type of aircraft."]


class FGVCAircraft(DatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            for line in f.readlines():
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)

        self._train_full = train
        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def train_full(self):
        """Full (non-few-shot) labeled train list, useful for fine-tuning."""
        return self._train_full

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            for line in f.readlines():
                parts = line.strip().split(" ")
                imname = parts[0] + ".jpg"
                classname = " ".join(parts[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                items.append(Datum(impath=impath, label=label, classname=classname))

        return items

    # ---------------------------------------------------------------- #
    # Preparation helpers (download only — FGVC ships its own splits)
    # ---------------------------------------------------------------- #

    @classmethod
    def auto_prepare(cls, root, seed: int = 1, force_generate_split: bool = False):
        """Ensure `<root>/fgvc_aircraft/{images/, variants.txt,
        images_variant_{train,val,test}.txt}` exist on disk; download via
        torchvision as needed.

        `seed` and `force_generate_split` are accepted for API parity with
        the other dataset modules but are ignored — FGVC Aircraft uses its
        own canonical split files, so there is no random partitioning step.

        Returns the prepared dataset root (`<root>/fgvc_aircraft`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        ds_root.mkdir(parents=True, exist_ok=True)

        if not cls._is_prepared(ds_root):
            cls._download_images(root)
        else:
            print(f"[fgvc_aircraft] dataset already present: {ds_root}")

        if not cls._is_prepared(ds_root):
            raise FileNotFoundError(
                f"FGVC Aircraft was not found under {ds_root} and could not "
                f"be downloaded automatically. Expected layout:\n"
                f"  {ds_root}/images/<id>.jpg\n"
                f"  {ds_root}/variants.txt\n"
                f"  {ds_root}/images_variant_train.txt\n"
                f"  {ds_root}/images_variant_val.txt\n"
                f"  {ds_root}/images_variant_test.txt\n"
                f"Grab the dataset from https://www.robots.ox.ac.uk/~vgg/data/"
                f"fgvc-aircraft/ or any mirror and unpack it there."
            )

        return ds_root

    @staticmethod
    def _is_prepared(ds_root: Path) -> bool:
        required = [
            ds_root / "images",
            ds_root / "variants.txt",
            ds_root / "images_variant_train.txt",
            ds_root / "images_variant_val.txt",
            ds_root / "images_variant_test.txt",
        ]
        return all(p.exists() for p in required)

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.FGVCAircraft to handle the tar download,
        then re-shape the unpack into Tip-Adapter's expected layout.

        torchvision unpacks to:
            <root>/fgvc-aircraft-2013b/data/images/<id>.jpg
            <root>/fgvc-aircraft-2013b/data/variants.txt
            <root>/fgvc-aircraft-2013b/data/images_variant_train.txt
            <root>/fgvc-aircraft-2013b/data/images_variant_val.txt
            <root>/fgvc-aircraft-2013b/data/images_variant_test.txt
        We re-layout that into:
            <root>/fgvc_aircraft/{images/, variants.txt, images_variant_*.txt}
        """
        import torchvision.datasets as tv_datasets

        print(f"[fgvc_aircraft] downloading FGVC Aircraft via torchvision into {root} ...")
        for split in ("train", "val", "test"):
            try:
                tv_datasets.FGVCAircraft(
                    root=str(root),
                    split=split,
                    annotation_level="variant",
                    download=True,
                )
            except Exception as exc:
                print(f"[fgvc_aircraft] torchvision download for split={split} failed: {exc}")

        tv_data = root / "fgvc-aircraft-2013b" / "data"
        tip_root = root / "fgvc_aircraft"
        if not tv_data.exists():
            return

        tip_root.mkdir(parents=True, exist_ok=True)

        # Move images/ directory.
        tip_images = tip_root / "images"
        if (tv_data / "images").exists() and not tip_images.exists():
            print(f"[fgvc_aircraft] moving {tv_data/'images'} -> {tip_images}")
            shutil.move(str(tv_data / "images"), str(tip_images))

        # Copy the variants list + the three split files.
        for fname in (
            "variants.txt",
            "images_variant_train.txt",
            "images_variant_val.txt",
            "images_variant_test.txt",
        ):
            src = tv_data / fname
            dst = tip_root / fname
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))
