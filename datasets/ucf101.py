"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/ucf101.py

UCF101 is a video-action recognition benchmark; CoOp / Tip-Adapter use the
mid-frame of each clip as a still image, hosted as a pre-extracted JPG
archive on Google Drive. The on-disk layout that Tip-Adapter expects is:

    <root>/ucf101/UCF-101-midframes/<Action_Underscored>/<image>.jpg
    <root>/ucf101/split_zhou_UCF101.json

The `auto_prepare` classmethod added here:
  * tries to download both the mid-frame archive and the official CoOp
    split JSON via gdown,
  * defensively unpacks the archive if it landed as a .zip,
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split when the official split JSON is unavailable but the mid-frame
    images are already on disk,
  * gives a clear "please drop the mid-frame folder here" error message
    when neither the archive nor a pre-existing tree can be found.

NOTE: there is no public torchvision auto-download for the mid-frame JPGs.
If gdown is blocked, manually unzip `UCF-101-midframes.zip` from CoOp's
shared Drive folder into `<root>/ucf101/` and re-run the prep script.
"""

import os
import random
import shutil
import zipfile
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, listdir_nohidden, read_json, write_json


template = ["a photo of a person doing {}."]

# Official CoOp split + mid-frame archive (both in CoOp's shared Drive folder).
# These IDs match the files referenced from CoOp's DATASETS.md walkthrough.
OFFICIAL_SPLIT_GDRIVE_ID = "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y"
MIDFRAMES_GDRIVE_ID = "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O"


class UCF101(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")

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
        """Ensure `<root>/ucf101/UCF-101-midframes/...` and the split JSON
        exist on disk; download / generate as needed.

        Returns the prepared dataset root (`<root>/ucf101`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "UCF-101-midframes"
        split_path = ds_root / "split_zhou_UCF101.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not cls._has_class_folders(image_dir):
            cls._download_images(ds_root)
        else:
            print(f"[ucf101] mid-frame images already present: {image_dir}")

        if not image_dir.exists() or not cls._has_class_folders(image_dir):
            raise FileNotFoundError(
                f"UCF-101 mid-frames were not found at {image_dir} and could "
                f"not be auto-downloaded. Drop the unzipped "
                f"`UCF-101-midframes/` folder into {ds_root}/ (from CoOp's "
                f"shared Drive — `UCF-101-midframes.zip`) and re-run this "
                f"script."
            )

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, split_path, seed=seed)
        else:
            print(f"[ucf101] split already present: {split_path}")

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

    @classmethod
    def _download_images(cls, ds_root: Path):
        """Try to fetch the mid-frame zip via gdown. Best-effort — the file
        is hosted on a CoOp-owned shared Drive and gdown sometimes hits
        anti-abuse rate-limits."""
        try:
            import gdown
        except ImportError:
            print("[ucf101] gdown not installed; cannot auto-download mid-frames")
            return

        zip_path = ds_root / "UCF-101-midframes.zip"
        url = f"https://drive.google.com/uc?id={MIDFRAMES_GDRIVE_ID}"
        print(f"[ucf101] trying mid-frames download via gdown: {url}")
        try:
            gdown.download(url, str(zip_path), quiet=False)
        except Exception as exc:
            print(f"[ucf101] mid-frames download failed ({exc})")
            return

        if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
            print(f"[ucf101] downloaded file at {zip_path} looks too small; "
                  f"the Drive quota may have been exceeded.")
            return

        try:
            print(f"[ucf101] extracting {zip_path} -> {ds_root}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(str(ds_root))
        except Exception as exc:
            print(f"[ucf101] zip extraction failed: {exc}")
            return

        # Normalize the unpack location: some archives wrap the tree in an
        # extra top-level folder.
        target = ds_root / "UCF-101-midframes"
        if not target.exists():
            for candidate in ds_root.rglob("UCF-101-midframes"):
                if candidate.is_dir() and candidate != target:
                    shutil.move(str(candidate), str(target))
                    break

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp UCF101 split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[ucf101] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[ucf101] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[ucf101] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[ucf101] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(image_dir: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects."""
        print(f"[ucf101] generating deterministic split (seed={seed})")
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
        print(f"[ucf101] wrote split ({len(categories)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
