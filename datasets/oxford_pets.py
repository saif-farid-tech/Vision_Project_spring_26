"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/oxford_pets.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.OxfordIIITPet` download (gives us
    `images/`, `annotations/trainval.txt`, and `annotations/test.txt`),
  * re-shapes the torchvision unpack into the Tip-Adapter layout
    `<root>/oxford_pets/{images/, annotations/, split_zhou_OxfordPets.json}`,
  * tries to fetch the official CoOp split JSON via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split (built off the `trainval.txt` + `test.txt` annotation files) if
    gdown is unavailable or the download fails.

The OxfordPets `read_split` / `save_split` / `split_trainval` helpers are
shared across all Tip-Adapter datasets, so they live here even though this
branch only consumes them through the OxfordPets loader itself.
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .utils import Datum, DatasetBase, read_json, write_json


template = ["a photo of a {}, a type of pet."]

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTZkw"


class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        self.template = template

        train, val, test = self.read_split(self.split_path, self.image_dir)
        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)

        self._train_full = train
        super().__init__(train_x=train_x, val=val, test=test)

    @property
    def train_full(self):
        """Full (non-few-shot) labeled train list, useful for fine-tuning."""
        return self._train_full

    # ---------------------------------------------------------------- #
    # Tip-Adapter helpers (shared across datasets in this package)
    # ---------------------------------------------------------------- #

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    # ---------------------------------------------------------------- #
    # Preparation helpers (download + split generation)
    # ---------------------------------------------------------------- #

    @classmethod
    def auto_prepare(cls, root, seed: int = 1, force_generate_split: bool = False):
        """Ensure `<root>/oxford_pets/{images, annotations,
        split_zhou_OxfordPets.json}` exist on disk; download / generate as
        needed.

        Returns the prepared dataset root (`<root>/oxford_pets`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "images"
        anno_dir = ds_root / "annotations"
        split_path = ds_root / "split_zhou_OxfordPets.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not anno_dir.exists():
            cls._download_images(root)
        else:
            print(f"[oxford_pets] images already present: {image_dir}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(ds_root, split_path, seed=seed)
        else:
            print(f"[oxford_pets] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.OxfordIIITPet to handle the tar download,
        then re-shape the unpack into Tip-Adapter's expected layout.

        torchvision unpacks to:
            <root>/oxford-iiit-pet/images/<image>.jpg
            <root>/oxford-iiit-pet/annotations/{trainval.txt,test.txt,...}
        We move that into:
            <root>/oxford_pets/{images,annotations}/...
        """
        import torchvision.datasets as tv_datasets

        print(f"[oxford_pets] downloading Oxford-IIIT Pet via torchvision into {root} ...")
        # Trigger both splits so trainval.txt + test.txt are extracted.
        for split in ("trainval", "test"):
            try:
                tv_datasets.OxfordIIITPet(
                    root=str(root), split=split, download=True,
                )
            except Exception as exc:
                print(f"[oxford_pets] torchvision download for split={split} failed: {exc}")

        tv_root = root / "oxford-iiit-pet"
        tip_root = root / "oxford_pets"
        tip_root.mkdir(parents=True, exist_ok=True)

        for sub in ("images", "annotations"):
            src = tv_root / sub
            dst = tip_root / sub
            if src.exists() and not dst.exists():
                print(f"[oxford_pets] moving {src} -> {dst}")
                shutil.move(str(src), str(dst))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp Oxford Pets split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[oxford_pets] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[oxford_pets] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[oxford_pets] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[oxford_pets] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _generate_split(ds_root: Path, split_path: Path, seed: int = 1):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects.

        Reads `annotations/trainval.txt` + `annotations/test.txt` to recover
        the (image, label, breed) tuples — the trainval and test rows are
        pooled before being re-split.
        """
        print(f"[oxford_pets] generating deterministic split (seed={seed})")
        image_dir = ds_root / "images"
        anno_dir = ds_root / "annotations"

        all_items = []
        for fname in ("trainval.txt", "test.txt"):
            path = anno_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"annotation file {path} is missing — the Oxford-IIIT Pet "
                    f"annotations/ archive must be present before generating "
                    f"the split."
                )
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    imname, label_str, _species, _ = line.split(" ")
                    label = int(label_str) - 1  # 0-based
                    breed = "_".join(imname.split("_")[:-1]).lower()
                    rel = f"{imname}.jpg"
                    if (image_dir / rel).exists():
                        all_items.append((rel, label, breed))

        rng = random.Random(seed)
        by_class = defaultdict(list)
        for rel, label, breed in all_items:
            by_class[(label, breed)].append(rel)

        split = {"train": [], "val": [], "test": []}
        for (label, breed) in sorted(by_class.keys()):
            paths = sorted(by_class[(label, breed)])
            rng.shuffle(paths)
            n = len(paths)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
            for rel in paths[:n_train]:
                split["train"].append([rel, label, breed])
            for rel in paths[n_train:n_train + n_val]:
                split["val"].append([rel, label, breed])
            for rel in paths[n_train + n_val:n_train + n_val + n_test]:
                split["test"].append([rel, label, breed])

        write_json(split, str(split_path))
        print(f"[oxford_pets] wrote split ({len(by_class)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
