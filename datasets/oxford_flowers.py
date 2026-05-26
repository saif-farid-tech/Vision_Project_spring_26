"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/oxford_flowers.py

Adds an `auto_prepare` classmethod that:
  * triggers a `torchvision.datasets.Flowers102` download (gives us
    `imagelabels.mat` and the per-image jpgs),
  * re-shapes the unpack into the Tip-Adapter layout
    `<root>/oxford_flowers/{jpg/, imagelabels.mat, cat_to_name.json,
    split_zhou_OxfordFlowers.json}`,
  * tries to fetch the official CoOp split + cat_to_name.json via gdown, and
  * falls back to a deterministic locally-generated 50/20/30 train/val/test
    split + a hardcoded class-name table if gdown is unavailable or fails.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json, write_json


template = ["a photo of a {}, a type of flower."]

# Official CoOp split (also used by Tip-Adapter), hosted on Google Drive.
OFFICIAL_SPLIT_GDRIVE_ID = "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT"
# Mirror of the standard cat_to_name.json mapping (Oxford Flowers 102).
CAT_TO_NAME_GDRIVE_ID = "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0"


# Fallback class-name table — the standard Oxford Flowers 102 mapping.
# Index 0 corresponds to the dataset's class id "1", and so on through "102".
CAT_TO_NAME_FALLBACK = {
    "1":  "pink primrose",
    "2":  "hard-leaved pocket orchid",
    "3":  "canterbury bells",
    "4":  "sweet pea",
    "5":  "english marigold",
    "6":  "tiger lily",
    "7":  "moon orchid",
    "8":  "bird of paradise",
    "9":  "monkshood",
    "10": "globe thistle",
    "11": "snapdragon",
    "12": "colt's foot",
    "13": "king protea",
    "14": "spear thistle",
    "15": "yellow iris",
    "16": "globe-flower",
    "17": "purple coneflower",
    "18": "peruvian lily",
    "19": "balloon flower",
    "20": "giant white arum lily",
    "21": "fire lily",
    "22": "pincushion flower",
    "23": "fritillary",
    "24": "red ginger",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "27": "prince of wales feathers",
    "28": "stemless gentian",
    "29": "artichoke",
    "30": "sweet william",
    "31": "carnation",
    "32": "garden phlox",
    "33": "love in the mist",
    "34": "mexican aster",
    "35": "alpine sea holly",
    "36": "ruby-lipped cattleya",
    "37": "cape flower",
    "38": "great masterwort",
    "39": "siam tulip",
    "40": "lenten rose",
    "41": "barbeton daisy",
    "42": "daffodil",
    "43": "sword lily",
    "44": "poinsettia",
    "45": "bolero deep blue",
    "46": "wallflower",
    "47": "marigold",
    "48": "buttercup",
    "49": "oxeye daisy",
    "50": "common dandelion",
    "51": "petunia",
    "52": "wild pansy",
    "53": "primula",
    "54": "sunflower",
    "55": "pelargonium",
    "56": "bishop of llandaff",
    "57": "gaura",
    "58": "geranium",
    "59": "orange dahlia",
    "60": "pink-yellow dahlia",
    "61": "cautleya spicata",
    "62": "japanese anemone",
    "63": "black-eyed susan",
    "64": "silverbush",
    "65": "californian poppy",
    "66": "osteospermum",
    "67": "spring crocus",
    "68": "bearded iris",
    "69": "windflower",
    "70": "tree poppy",
    "71": "gazania",
    "72": "azalea",
    "73": "water lily",
    "74": "rose",
    "75": "thorn apple",
    "76": "morning glory",
    "77": "passion flower",
    "78": "lotus",
    "79": "toad lily",
    "80": "anthurium",
    "81": "frangipani",
    "82": "clematis",
    "83": "hibiscus",
    "84": "columbine",
    "85": "desert-rose",
    "86": "tree mallow",
    "87": "magnolia",
    "88": "cyclamen",
    "89": "watercress",
    "90": "canna lily",
    "91": "hippeastrum",
    "92": "bee balm",
    "93": "ball moss",
    "94": "foxglove",
    "95": "bougainvillea",
    "96": "camellia",
    "97": "mallow",
    "98": "mexican petunia",
    "99": "bromelia",
    "100": "blanket flower",
    "101": "trumpet creeper",
    "102": "blackberry lily",
}


class OxfordFlowers(DatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, root, num_shots=-1):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_x = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # Keep the full train list around so downstream code can use it
        # for full-data fine-tuning.
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
        """Ensure `<root>/oxford_flowers/{jpg/, imagelabels.mat,
        cat_to_name.json, split_zhou_OxfordFlowers.json}` exist on disk;
        download / generate as needed.

        Returns the prepared dataset root (`<root>/oxford_flowers`).
        """
        root = Path(root)
        ds_root = root / cls.dataset_dir
        image_dir = ds_root / "jpg"
        labels_path = ds_root / "imagelabels.mat"
        catmap_path = ds_root / "cat_to_name.json"
        split_path = ds_root / "split_zhou_OxfordFlowers.json"

        ds_root.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not labels_path.exists():
            cls._download_images(root)
        else:
            print(f"[oxford_flowers] images already present: {image_dir}")

        if not catmap_path.exists():
            ok = cls._try_download_cat_to_name(catmap_path)
            if not ok:
                cls._write_fallback_cat_to_name(catmap_path)
        else:
            print(f"[oxford_flowers] cat_to_name.json already present: {catmap_path}")

        if force_generate_split or not split_path.exists():
            if not force_generate_split:
                ok = cls._try_download_official_split(split_path)
            else:
                ok = False
            if not ok:
                cls._generate_split(image_dir, labels_path, catmap_path,
                                    split_path, seed=seed)
        else:
            print(f"[oxford_flowers] split already present: {split_path}")

        return ds_root

    @staticmethod
    def _download_images(root: Path):
        """Use torchvision.datasets.Flowers102 to download images +
        imagelabels.mat (and setid.mat), then re-shape into Tip-Adapter's
        expected `oxford_flowers/jpg/...` layout.

        torchvision unpacks to:
            <root>/flowers-102/jpg/image_XXXXX.jpg
            <root>/flowers-102/imagelabels.mat
            <root>/flowers-102/setid.mat
        """
        import torchvision.datasets as tv_datasets

        print(f"[oxford_flowers] downloading Flowers102 via torchvision into {root} ...")
        tv_datasets.Flowers102(root=str(root), split="train", download=True)

        tv_root = root / "flowers-102"
        tip_root = root / "oxford_flowers"
        if not tv_root.exists():
            raise FileNotFoundError(
                f"torchvision did not create the expected directory {tv_root}"
            )

        tip_root.mkdir(parents=True, exist_ok=True)

        # Move jpg/ directory
        tip_jpg = tip_root / "jpg"
        tv_jpg = tv_root / "jpg"
        if tv_jpg.exists() and not tip_jpg.exists():
            print(f"[oxford_flowers] moving {tv_jpg} -> {tip_jpg}")
            shutil.move(str(tv_jpg), str(tip_jpg))
        elif tip_jpg.exists():
            print(f"[oxford_flowers] {tip_jpg} already exists, skipping move")

        # Move imagelabels.mat and setid.mat
        for fname in ("imagelabels.mat", "setid.mat"):
            src = tv_root / fname
            dst = tip_root / fname
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))

    @staticmethod
    def _try_download_official_split(split_path: Path) -> bool:
        """Try to fetch the CoOp Flowers102 split JSON via gdown."""
        try:
            import gdown
        except ImportError:
            print("[oxford_flowers] gdown not installed; will generate split locally")
            return False

        url = f"https://drive.google.com/uc?id={OFFICIAL_SPLIT_GDRIVE_ID}"
        try:
            print(f"[oxford_flowers] trying official split download via gdown: {url}")
            gdown.download(url, str(split_path), quiet=False)
        except Exception as exc:
            print(f"[oxford_flowers] official split download failed ({exc}); "
                  f"will generate locally")
            return False

        if split_path.exists() and split_path.stat().st_size > 0:
            print(f"[oxford_flowers] official split saved to {split_path}")
            return True
        return False

    @staticmethod
    def _try_download_cat_to_name(catmap_path: Path) -> bool:
        """Try to fetch cat_to_name.json via gdown."""
        try:
            import gdown
        except ImportError:
            return False

        url = f"https://drive.google.com/uc?id={CAT_TO_NAME_GDRIVE_ID}"
        try:
            print(f"[oxford_flowers] trying cat_to_name.json download via gdown: {url}")
            gdown.download(url, str(catmap_path), quiet=False)
        except Exception as exc:
            print(f"[oxford_flowers] cat_to_name download failed ({exc}); "
                  f"falling back to hardcoded table")
            return False

        if catmap_path.exists() and catmap_path.stat().st_size > 0:
            return True
        return False

    @staticmethod
    def _write_fallback_cat_to_name(catmap_path: Path):
        """Write the hardcoded CAT_TO_NAME_FALLBACK table to disk."""
        print(f"[oxford_flowers] writing fallback cat_to_name.json to {catmap_path}")
        with open(catmap_path, "w") as f:
            json.dump(CAT_TO_NAME_FALLBACK, f, indent=2)

    @staticmethod
    def _generate_split(
        image_dir: Path,
        label_file: Path,
        catmap_file: Path,
        split_path: Path,
        seed: int = 1,
    ):
        """Generate a deterministic per-class 50/20/30 train/val/test split
        in the same JSON format that OxfordPets.read_split expects."""
        from scipy.io import loadmat

        print(f"[oxford_flowers] generating deterministic split (seed={seed})")
        lab2cname = read_json(str(catmap_file))
        labels = loadmat(str(label_file))["labels"][0]

        tracker = defaultdict(list)
        for i, label in enumerate(labels):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            tracker[int(label)].append(imname)

        rng = random.Random(seed)
        split = {"train": [], "val": [], "test": []}
        for label in sorted(tracker.keys()):
            imnames = sorted(tracker[label])
            rng.shuffle(imnames)
            n = len(imnames)
            n_train = max(1, round(n * 0.5))
            n_val = max(1, round(n * 0.2))
            n_test = max(1, n - n_train - n_val)
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
            cname = lab2cname.get(str(label), f"class_{label}")
            zero_label = label - 1  # convert to 0-based
            for fname in imnames[:n_train]:
                split["train"].append([fname, zero_label, cname])
            for fname in imnames[n_train:n_train + n_val]:
                split["val"].append([fname, zero_label, cname])
            for fname in imnames[n_train + n_val:n_train + n_val + n_test]:
                split["test"].append([fname, zero_label, cname])

        write_json(split, str(split_path))
        print(f"[oxford_flowers] wrote split ({len(tracker)} classes, "
              f"{len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")
