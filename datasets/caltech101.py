"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/caltech101.py

Adds an `auto_prepare` classmethod that:
  * triggers a torchvision download of Caltech-101 if `101_ObjectCategories/`
    is missing,
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split if gdown is unavailable or the download fails.

The on-disk layout matches Tip-Adapter:
    <root>/caltech-101/101_ObjectCategories/<class>/<image>.jpg
    <root>/caltech-101/split_zhou_Caltech101.json
"""

import os
import random
import shutil
from pathlib import Path

from .utils import Datum, DatasetBase, listdir_nohidden, read_json, write_json
from .oxford_pets import OxfordPets


# CoOp / Tip-Adapter exclude these two folders from the 100-class split.
IGNORED = ["BACKGROUND_Google", "Faces_easy"]

# CoOp renames a few classes for cleaner natural-language prompts.
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN"

template = ["a photo of a {}."]


class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # Keep the full train list around so downstream code can use it
        # for full-data fine-tuning (the few-shot version is only useful
        # for the Tip-Adapter CLIP-prompt-learning regime).
        self._train_full = train

        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def train_full(self):
        """Full (non-few-shot) labeled train list, useful for fine-tuning."""
        return self._train_full

    # ---------------------------------------------------------------- #
    # Preparation helpers (download + split generation)
    # ---------------------------------------------------------------- #

    @classmethod
    def auto_prepare(cls, root, seed: int = 1, force_generate_split: bool = False):
        """Make sure `<root>/caltech-101/101_ObjectCategories/` and the split
        JSON exist on disk; download / generate as needed.

        Returns the prepared dataset root (`<root>/caltech-101`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "101_ObjectCategories"
        split_path = ds_root / "split_zhou_Caltech101.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            cls._download_images(root)
        else:
            print(f"[caltech101] images already present: {image_dir}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, split_path, seed=seed)
        else:
            print(f"[caltech101] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.Caltech101 to handle the tar download,
        then re-shape the unpacked layout into Tip-Adapter's expected one.

        torchvision unpacks to:
            <root>/caltech101/101_ObjectCategories/<class>/<id>.jpg
        We move that into:
            <root>/caltech-101/101_ObjectCategories/...
        """
        import torchvision.datasets as tv_datasets

        print(f"[caltech101] downloading Caltech-101 via torchvision into {root} ...")
        # torchvision will create <root>/caltech101/...
        tv_datasets.Caltech101(root=str(root), download=True)

        tv_root = root / "caltech101" / "101_ObjectCategories"
        tip_root = root / "caltech-101" / "101_ObjectCategories"
        if not tv_root.exists():
            raise FileNotFoundError(
                f"torchvision did not create the expected directory {tv_root}"
            )

        tip_root.parent.mkdir(parents=True, exist_ok=True)
        if tip_root.exists():
            print(f"[caltech101] {tip_root} already exists, skipping move")
            return

        print(f"[caltech101] moving {tv_root} -> {tip_root}")
        shutil.move(str(tv_root), str(tip_root))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp Caltech-101 split JSON via gdown.
        Returns True on success, False otherwise (caller then generates)."""
        try:
            import gdown
        except ImportError:
            print("[caltech101] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[caltech101] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[caltech101] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[caltech101] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(image_dir: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects.

        Classes in IGNORED are excluded; class names in NEW_CNAMES are
        relabelled. Within each kept class, files are shuffled with the
        given seed and then split 50% train / 20% val / 30% test (rounded
        so every class gets at least 1 sample in each set).
        """
        print(f"[caltech101] generating deterministic split (seed={seed})")
        categories = [
            d for d in listdir_nohidden(str(image_dir), sort=True)
            if d not in IGNORED
        ]
        rng = random.Random(seed)

        split = {"train": [], "val": [], "test": []}
        for label, category in enumerate(categories):
            classname = NEW_CNAMES.get(category, category)
            cls_dir = image_dir / category
            files = sorted([f for f in os.listdir(cls_dir)
                            if not f.startswith(".")])
            rng.shuffle(files)

            n = len(files)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            # Ensure test gets the remainder (at least 1).
            n_test = max(1, n - n_train - n_val)
            # Re-balance if rounding overflowed.
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test

            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:n_train + n_val + n_test]

            for fname in train_files:
                split["train"].append([f"{category}/{fname}", label, classname])
            for fname in val_files:
                split["val"].append([f"{category}/{fname}", label, classname])
            for fname in test_files:
                split["test"].append([f"{category}/{fname}", label, classname])

        write_json(split, str(split_path))
        print(f"[caltech101] wrote split ({len(categories)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
