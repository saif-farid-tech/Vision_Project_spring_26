"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/stanford_cars.py

Adds an `auto_prepare` classmethod that:
  * tries to download Stanford Cars via `torchvision.datasets.StanfordCars`
    and re-layouts the unpack into the Tip-Adapter convention
    `<root>/stanford_cars/{cars_train, cars_test, devkit, ...}`,
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split built from `cars_meta.mat` + `cars_train_annos.mat` +
    `cars_test_annos_withlabels.mat` if gdown is unavailable.

NOTE on data availability: as of late 2022 the original Stanford Cars host
(`ai.stanford.edu/~jkrause/cars/`) is offline; torchvision's StanfordCars
download will fail accordingly. If the auto-download fails, you can grab the
dataset from any well-known mirror (Kaggle, HuggingFace) and unpack it under
`<root>/stanford_cars/` with the expected layout — then re-run this prep
step (it will skip the download and just (re)build the split).
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json, write_json


template = ["a photo of a {}."]

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT"


class StanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
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
        """Ensure `<root>/stanford_cars/{...}` and the split JSON exist on
        disk; download / generate as needed.

        Returns the prepared dataset root (`<root>/stanford_cars`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        split_path = ds_root / "split_zhou_StanfordCars.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not cls._has_images(ds_root):
            cls._download_images(root)
        else:
            print(f"[stanford_cars] images already present: {ds_root}")

        if not cls._has_images(ds_root):
            raise FileNotFoundError(
                f"Stanford Cars images were not found under {ds_root} and "
                f"could not be downloaded automatically. The original host "
                f"(ai.stanford.edu) is offline; please download the dataset "
                f"from a mirror (Kaggle / HuggingFace) and unpack it into "
                f"{ds_root}/ so it has cars_train/, cars_test/, and "
                f"devkit/cars_meta.mat + devkit/cars_train_annos.mat + "
                f"cars_test_annos_withlabels.mat. Then re-run this script."
            )

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(ds_root, split_path, seed=seed)
        else:
            print(f"[stanford_cars] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _has_images(ds_root: Path) -> bool:
        """Loosely verify that the Stanford Cars unpack is on disk."""
        train_imgs = list((ds_root / "cars_train").glob("*.jpg")) if (ds_root / "cars_train").exists() else []
        test_imgs = list((ds_root / "cars_test").glob("*.jpg")) if (ds_root / "cars_test").exists() else []
        return len(train_imgs) > 0 and len(test_imgs) > 0

    @staticmethod
    def _download_images(root: Path):
        """Try downloading via torchvision.datasets.StanfordCars.

        The original host is offline so this will likely fail; we still
        attempt it because torchvision may eventually pick up a mirror, and
        because the same call will succeed (no-op) if the user has dropped
        a pre-extracted copy at the canonical path.
        """
        try:
            import torchvision.datasets as tv_datasets
        except ImportError:
            print("[stanford_cars] torchvision missing; cannot auto-download")
            return

        print(f"[stanford_cars] attempting Stanford Cars download via torchvision into {root} ...")
        tv_root = root / "stanford_cars"
        tv_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            try:
                tv_datasets.StanfordCars(root=str(root), split=split, download=True)
            except Exception as exc:
                print(f"[stanford_cars] torchvision download for split={split} failed: {exc}")

        # torchvision lays things out under <root>/stanford_cars/ already, so
        # there is nothing to move when it succeeds. If a sibling layout
        # was used (older torchvision versions), normalize it here.
        legacy = root / "cars"
        if legacy.exists() and not (tv_root / "cars_train").exists():
            for sub in ("cars_train", "cars_test", "devkit"):
                src = legacy / sub
                dst = tv_root / sub
                if src.exists() and not dst.exists():
                    shutil.move(str(src), str(dst))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp Stanford Cars split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[stanford_cars] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[stanford_cars] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[stanford_cars] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[stanford_cars] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(ds_root: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 split using the
        canonical Stanford Cars meta + annotation .mat files.

        Expected files (any one valid layout):
            <ds_root>/devkit/cars_meta.mat
            <ds_root>/devkit/cars_train_annos.mat
            <ds_root>/cars_test_annos_withlabels.mat
        """
        from scipy.io import loadmat

        print(f"[stanford_cars] generating deterministic split (seed={seed})")

        # Locate the meta + annotation files (torchvision uses a slightly
        # different path than the original release).
        candidates = [
            ds_root / "devkit" / "cars_meta.mat",
            ds_root / "cars_meta.mat",
        ]
        meta_path = next((p for p in candidates if p.exists()), None)
        if meta_path is None:
            raise FileNotFoundError(
                f"cars_meta.mat not found under {ds_root}. "
                f"Provide the Stanford Cars devkit alongside the images."
            )

        train_ann_path = next(
            (p for p in [ds_root / "devkit" / "cars_train_annos.mat",
                         ds_root / "cars_train_annos.mat"] if p.exists()),
            None,
        )
        test_ann_path = next(
            (p for p in [ds_root / "cars_test_annos_withlabels.mat",
                         ds_root / "devkit" / "cars_test_annos_withlabels.mat"] if p.exists()),
            None,
        )
        if train_ann_path is None or test_ann_path is None:
            raise FileNotFoundError(
                f"Annotation .mat files not found under {ds_root}. "
                f"Need cars_train_annos.mat and cars_test_annos_withlabels.mat."
            )

        class_names = [c[0] for c in loadmat(str(meta_path))["class_names"][0]]

        def _collect(ann_path, image_subdir):
            ann = loadmat(str(ann_path))["annotations"][0]
            items = []
            for row in ann:
                fname = row["fname"][0]
                label = int(row["class"][0, 0]) - 1  # 0-based
                items.append((f"{image_subdir}/{fname}", label))
            return items

        all_items = (
            _collect(train_ann_path, "cars_train")
            + _collect(test_ann_path, "cars_test")
        )

        # Group by class, shuffle deterministically, then split 50/20/30.
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for rel, label in all_items:
            by_class[label].append(rel)

        split = {"train": [], "val": [], "test": []}
        for label in sorted(by_class.keys()):
            paths = sorted(by_class[label])
            rng.shuffle(paths)
            n = len(paths)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
            cname = class_names[label]
            for rel in paths[:n_train]:
                split["train"].append([rel, label, cname])
            for rel in paths[n_train:n_train + n_val]:
                split["val"].append([rel, label, cname])
            for rel in paths[n_train + n_val:n_train + n_val + n_test]:
                split["test"].append([rel, label, cname])

        write_json(split, str(split_path))
        print(f"[stanford_cars] wrote split ({len(by_class)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
