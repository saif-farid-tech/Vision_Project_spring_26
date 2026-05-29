"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/eurosat.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.EuroSAT` download (gives us the RGB
    EuroSAT image tree with one folder per land-cover class),
  * makes sure the unpack is laid out as `<root>/eurosat/2750/<class>/...`
    (the canonical Tip-Adapter convention),
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split if gdown is unavailable or the download fails.

The on-disk layout matches Tip-Adapter:
    <root>/eurosat/2750/<RawClass>/<image>.jpg
    <root>/eurosat/split_zhou_EuroSAT.json
"""

import os
import random
import shutil
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, listdir_nohidden, read_json, write_json


template = ["a centered satellite photo of {}."]


# Tip-Adapter / CoOp rename the 10 raw class folders to natural-language
# phrases that play nicely with the CLIP prompt template above.
NEW_CNAMES = {
    "AnnualCrop":           "Annual Crop Land",
    "Forest":               "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway":              "Highway or Road",
    "Industrial":           "Industrial Buildings",
    "Pasture":              "Pasture Land",
    "PermanentCrop":        "Permanent Crop Land",
    "Residential":          "Residential Buildings",
    "River":                "River",
    "SeaLake":              "Sea or Lake",
}

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"


class EuroSAT(DatasetBase):

    dataset_dir = "eurosat"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)

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
        """Ensure `<root>/eurosat/2750/<class>/<image>.jpg` and the split
        JSON exist on disk; download / generate as needed.

        Returns the prepared dataset root (`<root>/eurosat`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "2750"
        split_path = ds_root / "split_zhou_EuroSAT.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not cls._has_class_folders(image_dir):
            cls._download_images(root)
        else:
            print(f"[eurosat] images already present: {image_dir}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, split_path, seed=seed)
        else:
            print(f"[eurosat] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _has_class_folders(image_dir: Path) -> bool:
        if not image_dir.exists():
            return False
        for sub in image_dir.iterdir():
            if sub.is_dir():
                try:
                    next(sub.glob("*.jpg"))
                    return True
                except StopIteration:
                    continue
        return False

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.EuroSAT to handle the zip download,
        then re-shape the unpack into Tip-Adapter's expected layout.

        torchvision unpacks to:
            <root>/eurosat/2750/<class>/<image>.jpg
        which already matches the Tip-Adapter convention, so the move step
        is typically a no-op; we keep it defensive in case torchvision
        changes its layout in a future release.
        """
        import torchvision.datasets as tv_datasets

        print(f"[eurosat] downloading EuroSAT via torchvision into {root} ...")
        try:
            tv_datasets.EuroSAT(root=str(root), download=True)
        except Exception as exc:
            print(f"[eurosat] torchvision EuroSAT download failed: {exc}")

        tip_root = root / "eurosat"
        target_images = tip_root / "2750"
        if target_images.exists() and EuroSAT._has_class_folders(target_images):
            return

        # Defensive fallback: look for any nested `2750/` folder under
        # <root>/eurosat/ and move it to the expected place.
        for candidate in tip_root.rglob("2750"):
            if candidate == target_images or not candidate.is_dir():
                continue
            if EuroSAT._has_class_folders(candidate):
                target_images.parent.mkdir(parents=True, exist_ok=True)
                print(f"[eurosat] moving {candidate} -> {target_images}")
                shutil.move(str(candidate), str(target_images))
                return

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp EuroSAT split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[eurosat] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[eurosat] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[eurosat] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[eurosat] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(image_dir: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects.

        Raw class folder names are remapped to NEW_CNAMES so the classnames
        recorded in the JSON match what Tip-Adapter / CoOp use.
        """
        print(f"[eurosat] generating deterministic split (seed={seed})")
        categories = sorted(p.name for p in image_dir.iterdir()
                            if p.is_dir() and not p.name.startswith("."))
        rng = random.Random(seed)

        split = {"train": [], "val": [], "test": []}
        for label, category in enumerate(categories):
            cls_dir = image_dir / category
            files = sorted([f for f in os.listdir(cls_dir)
                            if not f.startswith(".") and f.lower().endswith(".jpg")])
            rng.shuffle(files)

            n = len(files)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test

            classname = NEW_CNAMES.get(category, category)
            for fname in files[:n_train]:
                split["train"].append([f"{category}/{fname}", label, classname])
            for fname in files[n_train:n_train + n_val]:
                split["val"].append([f"{category}/{fname}", label, classname])
            for fname in files[n_train + n_val:n_train + n_val + n_test]:
                split["test"].append([f"{category}/{fname}", label, classname])

        write_json(split, str(split_path))
        print(f"[eurosat] wrote split ({len(categories)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
