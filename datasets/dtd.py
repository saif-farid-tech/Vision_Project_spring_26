"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/dtd.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.DTD` download (gives us the DTD image
    tree with one folder per texture category),
  * re-shapes the unpack into the Tip-Adapter layout
    `<root>/dtd/images/<category>/<image>.jpg`,
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split if gdown is unavailable or the download fails.

The on-disk layout matches Tip-Adapter:
    <root>/dtd/images/<category>/<image>.jpg
    <root>/dtd/split_zhou_DescribableTextures.json
"""

import os
import random
import shutil
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, listdir_nohidden, read_json, write_json


template = ["{} texture."]

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"


class DescribableTextures(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(
            self.dataset_dir, "split_zhou_DescribableTextures.json"
        )

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
        """Ensure `<root>/dtd/images/` and the split JSON exist on disk;
        download / generate as needed.

        Returns the prepared dataset root (`<root>/dtd`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "images"
        split_path = ds_root / "split_zhou_DescribableTextures.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not cls._has_class_folders(image_dir):
            cls._download_images(root)
        else:
            print(f"[dtd] images already present: {image_dir}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, split_path, seed=seed)
        else:
            print(f"[dtd] split already present: {split_path}")

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
        """Use torchvision.datasets.DTD to handle the tar download, then
        re-shape the unpacked layout into Tip-Adapter's expected one.

        torchvision unpacks to:
            <root>/dtd/dtd-r1.0.1/dtd/images/<category>/<image>.jpg
            <root>/dtd/dtd-r1.0.1/dtd/labels/...
            <root>/dtd/dtd-r1.0.1/dtd/imdb/...
        We move the images tree into:
            <root>/dtd/images/<category>/<image>.jpg
        """
        import torchvision.datasets as tv_datasets

        print(f"[dtd] downloading DTD via torchvision into {root} ...")
        # Trigger every split so all three label files are written.
        for split in ("train", "val", "test"):
            try:
                tv_datasets.DTD(
                    root=str(root), split=split, download=True,
                )
            except Exception as exc:
                print(f"[dtd] torchvision download for split={split} failed: {exc}")

        tip_root = root / "dtd"
        target_images = tip_root / "images"

        # Try the standard release layout first.
        candidates = [
            tip_root / "dtd-r1.0.1" / "dtd" / "images",
            tip_root / "dtd" / "images",
        ]
        source_images = next((p for p in candidates if p.exists()), None)
        if source_images is None or target_images.exists():
            if target_images.exists():
                print(f"[dtd] {target_images} already exists, skipping move")
            return

        target_images.parent.mkdir(parents=True, exist_ok=True)
        print(f"[dtd] moving {source_images} -> {target_images}")
        shutil.move(str(source_images), str(target_images))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp DTD split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[dtd] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[dtd] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[dtd] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[dtd] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(image_dir: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects."""
        print(f"[dtd] generating deterministic split (seed={seed})")
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

            for fname in files[:n_train]:
                split["train"].append([f"{category}/{fname}", label, category])
            for fname in files[n_train:n_train + n_val]:
                split["val"].append([f"{category}/{fname}", label, category])
            for fname in files[n_train + n_val:n_train + n_val + n_test]:
                split["test"].append([f"{category}/{fname}", label, category])

        write_json(split, str(split_path))
        print(f"[dtd] wrote split ({len(categories)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
