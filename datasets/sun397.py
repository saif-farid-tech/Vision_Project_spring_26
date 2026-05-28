"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/sun397.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.SUN397` download (gives us the SUN397
    image tree with letter-prefixed subdirectories),
  * re-shapes the unpack into the Tip-Adapter layout
    `<root>/sun397/SUN397/<letter>/<scene>/<image>.jpg`,
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split if gdown is unavailable or the download fails.

The on-disk layout matches Tip-Adapter:
    <root>/sun397/SUN397/<letter>/<scene>/<image>.jpg
    <root>/sun397/split_zhou_SUN397.json
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, listdir_nohidden, read_json, write_json


template = ["a photo of a {}."]

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1y2RD81BYuiyvebdN-JxPwFiwLNFKZTKf"


class SUN397(DatasetBase):

    dataset_dir = "sun397"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")

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
        """Ensure `<root>/sun397/SUN397/...` and the split JSON exist on
        disk; download / generate as needed.

        Returns the prepared dataset root (`<root>/sun397`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "SUN397"
        split_path = ds_root / "split_zhou_SUN397.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not cls._has_class_tree(image_dir):
            cls._download_images(root)
        else:
            print(f"[sun397] images already present: {image_dir}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, split_path, seed=seed)
        else:
            print(f"[sun397] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _has_class_tree(image_dir: Path) -> bool:
        """SUN397 has a letter-prefixed top level (a, b, ..., y) with class
        folders nested inside. Just verify at least one letter folder exists
        and that it contains image-bearing class folders."""
        if not image_dir.exists():
            return False
        for letter_dir in image_dir.iterdir():
            if letter_dir.is_dir() and len(letter_dir.name) == 1:
                for scene_dir in letter_dir.iterdir():
                    if scene_dir.is_dir():
                        try:
                            next(scene_dir.rglob("*.jpg"))
                            return True
                        except StopIteration:
                            continue
        return False

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.SUN397 to handle the tar download, then
        re-shape the unpacked layout into Tip-Adapter's expected one.

        torchvision unpacks to:
            <root>/SUN397/<letter>/<scene>/<image>.jpg
            <root>/ClassName.txt
        We move that into:
            <root>/sun397/SUN397/<letter>/<scene>/<image>.jpg
        """
        import torchvision.datasets as tv_datasets

        print(f"[sun397] downloading SUN397 via torchvision into {root} ...")
        try:
            tv_datasets.SUN397(root=str(root), download=True)
        except Exception as exc:
            print(f"[sun397] torchvision SUN397 download failed: {exc}")

        tv_root = root / "SUN397"
        tip_root = root / "sun397" / "SUN397"
        if not tv_root.exists():
            return

        tip_root.parent.mkdir(parents=True, exist_ok=True)
        if tip_root.exists() and SUN397._has_class_tree(tip_root):
            print(f"[sun397] {tip_root} already exists, skipping move")
            return

        print(f"[sun397] moving {tv_root} -> {tip_root}")
        shutil.move(str(tv_root), str(tip_root))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp SUN397 split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[sun397] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[sun397] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[sun397] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[sun397] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(image_dir: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split.

        Walks the letter-prefixed `<image_dir>/<letter>/<scene>/...` tree
        and builds a CoOp-style JSON whose image paths are relative to
        `image_dir`. Classname follows Tip-Adapter's convention of stripping
        the leading letter token and reversing the remaining path
        components (e.g. `/a/airfield` -> "airfield", `/a/airport_terminal`
        -> "airport terminal", `/c/canyon/indoor` -> "indoor canyon").
        """
        print(f"[sun397] generating deterministic split (seed={seed})")

        # Discover all class folders (deepest level that holds images).
        class_to_images = defaultdict(list)
        for letter_dir in sorted(image_dir.iterdir()):
            if not letter_dir.is_dir() or len(letter_dir.name) != 1:
                continue
            for scene_dir in sorted(letter_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                # Some scenes nest one extra level (indoor/outdoor variants).
                subscenes = [p for p in scene_dir.iterdir() if p.is_dir()]
                if subscenes:
                    for sub in sorted(subscenes):
                        rel_cls = sub.relative_to(image_dir).as_posix()
                        for img in sorted(sub.glob("*.jpg")):
                            class_to_images[rel_cls].append(img)
                else:
                    rel_cls = scene_dir.relative_to(image_dir).as_posix()
                    for img in sorted(scene_dir.glob("*.jpg")):
                        class_to_images[rel_cls].append(img)

        def _classname(rel_cls: str) -> str:
            # rel_cls looks like "a/airport_terminal" or "c/canyon/indoor".
            parts = rel_cls.split("/")[1:]   # drop leading letter
            parts = parts[::-1]              # reverse so indoor/outdoor comes first
            return " ".join(p.replace("_", " ") for p in parts)

        sorted_classes = sorted(class_to_images.keys())
        rng = random.Random(seed)
        split = {"train": [], "val": [], "test": []}
        for label, rel_cls in enumerate(sorted_classes):
            classname = _classname(rel_cls)
            paths = sorted(class_to_images[rel_cls])
            rng.shuffle(paths)
            n = len(paths)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test

            for p in paths[:n_train]:
                rel = p.relative_to(image_dir).as_posix()
                split["train"].append([rel, label, classname])
            for p in paths[n_train:n_train + n_val]:
                rel = p.relative_to(image_dir).as_posix()
                split["val"].append([rel, label, classname])
            for p in paths[n_train + n_val:n_train + n_val + n_test]:
                rel = p.relative_to(image_dir).as_posix()
                split["test"].append([rel, label, classname])

        write_json(split, str(split_path))
        print(f"[sun397] wrote split ({len(sorted_classes)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
