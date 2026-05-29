"""Vendored / extended from
https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/sun397.py

Adds an `auto_prepare` classmethod that:
  * downloads + extracts the SUN397 image archive when a reachable
    `SUN397.tar.gz` is available (`SUN397_URL` override or the Princeton URLs),
    falling back to the HuggingFace `datasets` loader (the official Princeton
    tarball was removed and now 404s) which materializes the images and writes
    the split JSON directly,
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

        if not cls._images_present(image_dir):
            # Prefer the (fast) tarball download; if every tarball source is
            # dead — the official Princeton URL now 404s — fall back to the
            # HuggingFace `datasets` loader, which materializes the images and
            # writes the split JSON in one go.
            if not cls._download_images(root):
                print("[sun397] falling back to the HuggingFace `datasets` "
                      "loader for SUN397 ...")
                cls._prepare_from_hf(image_dir, split_path, seed=seed)
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
    def _images_present(image_dir: Path) -> bool:
        """True if any SUN397 images already live under ``image_dir`` (either
        the canonical letter/scene tree or the flat per-class layout written
        by the HuggingFace fallback)."""
        if not image_dir.exists():
            return False
        try:
            next(image_dir.rglob("*.jpg"))
            return True
        except StopIteration:
            return False

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
    def _candidate_image_urls() -> list:
        """Ordered list of URLs to try for the SUN397 image archive.

        The upstream torchvision URL (``http://vision.princeton.edu/...``) now
        returns 404 — the official tarball was removed from that host, which is
        why ``download=True`` "just fails". A ``SUN397_URL`` override is honored
        first (point it at any reachable ``SUN397.tar.gz`` mirror), then the
        (now usually dead) Princeton URLs are tried. When all of these fail the
        caller falls back to the HuggingFace ``datasets`` loader, which is the
        reliable source for SUN397 today.
        """
        urls = []
        env_url = os.environ.get("SUN397_URL")
        if env_url:
            urls.append(env_url)
        urls += [
            "https://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
            "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
        ]
        # De-duplicate while preserving order.
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    @staticmethod
    def _download_images(root: Path) -> bool:
        """Try to download + extract the SUN397 image archive, then re-shape
        the unpacked layout into Tip-Adapter's expected one.

        The archive unpacks to:
            <root>/SUN397/<letter>/<scene>/<image>.jpg
            <root>/SUN397/ClassName.txt
        We move that into:
            <root>/sun397/SUN397/<letter>/<scene>/<image>.jpg

        Tries multiple URLs (HuggingFace mirror first, then Princeton), and
        falls back to skipping the MD5 verification if the checksum mismatches.

        Returns ``True`` if the images are now present in the Tip-Adapter
        layout, ``False`` if every tarball source failed (the caller then
        falls back to the HuggingFace ``datasets`` loader).
        """
        import torchvision.datasets as tv_datasets
        from torchvision.datasets.utils import download_and_extract_archive

        tv_root = root / "SUN397"
        tip_root = root / "sun397" / "SUN397"

        # Already moved into place by a previous run?
        if tip_root.exists() and SUN397._has_class_tree(tip_root):
            return True

        # Already unpacked (e.g. a previous interrupted run)? Skip download.
        if not (tv_root.exists() and SUN397._has_class_tree(tv_root)):
            expected_md5 = getattr(tv_datasets.SUN397, "_DATASET_MD5", None)
            for url in SUN397._candidate_image_urls():
                for check_md5 in (expected_md5, None):
                    if check_md5 is None and expected_md5 is not None:
                        print("[sun397] retrying download without MD5 "
                              "verification (checksum mismatch) ...")
                    try:
                        print(f"[sun397] downloading SUN397 archive from {url} "
                              f"into {root} (this is ~37 GB, may take a while) ...")
                        download_and_extract_archive(
                            url, download_root=str(root), md5=check_md5
                        )
                        break
                    except Exception as exc:
                        print(f"[sun397] download from {url} failed: {exc}")
                        # Only the md5-off retry helps for checksum errors; for
                        # anything else move on to the next URL.
                        msg = str(exc).lower()
                        if "md5" not in msg and "match" not in msg:
                            break
                if tv_root.exists() and SUN397._has_class_tree(tv_root):
                    break

            if not (tv_root.exists() and SUN397._has_class_tree(tv_root)):
                print("[sun397] all tarball sources failed.")
                return False

        tip_root.parent.mkdir(parents=True, exist_ok=True)
        if tip_root.exists() and SUN397._has_class_tree(tip_root):
            print(f"[sun397] {tip_root} already exists, skipping move")
            return True

        print(f"[sun397] moving {tv_root} -> {tip_root}")
        shutil.move(str(tv_root), str(tip_root))
        return True

    @staticmethod
    def _prepare_from_hf(image_dir: Path, split_path: Path, seed: int = 1):
        """Fallback path: load SUN397 through the HuggingFace ``datasets``
        library, write every image to ``<image_dir>/<NNN>/<file>.jpg`` (one
        folder per class index), and emit a CoOp-style ``split_zhou_SUN397``
        JSON directly.

        This is used when the tarball mirrors are unreachable (the official
        Princeton URL was removed and now 404s). The HF datasets are the
        reliable, canonical source for SUN397 today.

        Note: this repo ships a package literally named ``datasets`` (this
        very module lives in it), which shadows the HuggingFace ``datasets``
        library on ``sys.path``. We temporarily hide the vendored package so
        the real library can be imported, then restore it.
        """
        import importlib
        import subprocess
        import sys as _sys

        vendored_paths = [
            p for p in list(_sys.path)
            if os.path.isfile(os.path.join(p or os.getcwd(),
                                           "datasets", "sun397.py"))
        ]
        saved_modules = {
            k: _sys.modules.pop(k) for k in list(_sys.modules)
            if k == "datasets" or k.startswith("datasets.")
        }
        for p in vendored_paths:
            _sys.path.remove(p)
        importlib.invalidate_caches()
        try:
            try:
                hfds = importlib.import_module("datasets")
            except ModuleNotFoundError:
                hfds = None
            if hfds is None or not hasattr(hfds, "load_dataset"):
                print("[sun397] installing the HuggingFace `datasets` "
                      "library ...")
                subprocess.check_call(
                    [_sys.executable, "-m", "pip", "install", "-q", "datasets"]
                )
                for k in [k for k in list(_sys.modules)
                          if (k == "datasets" or k.startswith("datasets."))
                          and k not in saved_modules]:
                    del _sys.modules[k]
                importlib.invalidate_caches()
                hfds = importlib.import_module("datasets")
            if not hasattr(hfds, "load_dataset"):
                raise RuntimeError(
                    "[sun397] could not import the HuggingFace `datasets` "
                    "library — it is shadowed by the vendored datasets/ "
                    "package and the workaround failed. Run prepare from a "
                    "directory that is not the repo root, or `pip install "
                    "datasets` and retry.")
            SUN397._materialize_split_from_hf(
                hfds.load_dataset, hfds.Image, image_dir, split_path, seed)
        finally:
            for k in [k for k in list(_sys.modules)
                      if k == "datasets" or k.startswith("datasets.")]:
                del _sys.modules[k]
            _sys.modules.update(saved_modules)
            for p in reversed(vendored_paths):
                _sys.path.insert(0, p)
            importlib.invalidate_caches()

    @staticmethod
    def _materialize_split_from_hf(load_dataset, HFImage, image_dir: Path,
                                   split_path: Path, seed: int = 1):
        """Load SUN397 via an already-imported HuggingFace ``load_dataset``,
        write the images to disk under one folder per class index, and emit
        the CoOp-style split JSON."""
        repos = []
        env_repo = os.environ.get("SUN397_HF_REPO")
        if env_repo:
            repos.append(env_repo)
        repos += ["tanganke/sun397", "dpdl-benchmark/sun397"]

        dd = None
        used = None
        last_exc = None
        for repo in repos:
            try:
                print(f"[sun397] loading SUN397 from HuggingFace dataset "
                      f"'{repo}' ...")
                dd = load_dataset(repo)
                used = repo
                break
            except Exception as exc:
                last_exc = exc
                print(f"[sun397] load_dataset('{repo}') failed: {exc}")

        if dd is None:
            raise RuntimeError(
                "[sun397] could not obtain SUN397 from any source.\n"
                f"  Tried tarball URLs: {SUN397._candidate_image_urls()}\n"
                f"  Tried HuggingFace repos: {repos}\n"
                f"  Last error: {last_exc}\n"
                "  Fix: set SUN397_URL to a reachable SUN397.tar.gz mirror, or "
                "SUN397_HF_REPO to a loadable HuggingFace dataset id, or extract "
                f"the images manually so that '{image_dir}/<class>/<image>.jpg' "
                "exists, then re-run."
            )

        # Discover the image and label columns from the first available split.
        first_split = next(iter(dd.values()))
        feats = first_split.features
        img_col = next(
            (k for k, v in feats.items()
             if v.__class__.__name__ == "Image"), None
        )
        lbl_col = next(
            (k for k, v in feats.items()
             if v.__class__.__name__ == "ClassLabel"), None
        )
        if img_col is None:
            img_col = "image"
        if lbl_col is None:
            lbl_col = "label"

        label_names = getattr(feats.get(lbl_col, None), "names", None)

        def _classname(label: int) -> str:
            if label_names and 0 <= label < len(label_names):
                raw = str(label_names[label])
            else:
                raw = str(label)
            # Names may look like "/a/abbey" or "airport_terminal".
            raw = raw.strip("/").split("/")[-1]
            return raw.replace("_", " ").strip() or str(label)

        # Access raw encoded bytes instead of decoded PIL images, so we copy
        # the original JPEGs without a lossy re-encode.
        dd = dd.cast_column(img_col, HFImage(decode=False))

        image_dir.mkdir(parents=True, exist_ok=True)

        def _materialize(hf_split, tag: str):
            records = []
            for i, ex in enumerate(hf_split):
                label = int(ex[lbl_col])
                cell = ex[img_col]
                data = cell["bytes"] if isinstance(cell, dict) else None
                src_path = cell.get("path") if isinstance(cell, dict) else None
                cdir = image_dir / f"{label:03d}"
                cdir.mkdir(parents=True, exist_ok=True)
                fpath = cdir / f"{tag}_{i:06d}.jpg"
                if not fpath.exists():
                    if data is not None:
                        fpath.write_bytes(data)
                    elif src_path and os.path.exists(src_path):
                        shutil.copyfile(src_path, fpath)
                    else:
                        # Last resort: decode + re-encode.
                        from PIL import Image as PILImage
                        import io
                        PILImage.open(io.BytesIO(data)).convert("RGB").save(
                            fpath, "JPEG", quality=95)
                rel = fpath.relative_to(image_dir).as_posix()
                records.append([rel, label, _classname(label)])
            return records

        split = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)

        def _split_off_val(records, frac=0.2):
            by_label = defaultdict(list)
            for rec in records:
                by_label[rec[1]].append(rec)
            train, val = [], []
            for label in sorted(by_label):
                recs = by_label[label]
                rng.shuffle(recs)
                n_val = max(1, round(len(recs) * frac)) if len(recs) > 1 else 0
                val.extend(recs[:n_val])
                train.extend(recs[n_val:])
            return train, val

        n_total = 0
        if "test" in dd:
            print("[sun397] materializing HF 'test' split ...")
            split["test"] = _materialize(dd["test"], "test")
            n_total += len(split["test"])
            train_key = "train" if "train" in dd else next(
                (k for k in dd if k != "test"), None)
            if train_key is not None:
                print(f"[sun397] materializing HF '{train_key}' split ...")
                trainval = _materialize(dd[train_key], "train")
                n_total += len(trainval)
                split["train"], split["val"] = _split_off_val(trainval, 0.2)
        else:
            # Single split — carve a deterministic 50/20/30 partition.
            print("[sun397] materializing single HF split ...")
            allrecs = _materialize(first_split, "all")
            n_total += len(allrecs)
            by_label = defaultdict(list)
            for rec in allrecs:
                by_label[rec[1]].append(rec)
            for label in sorted(by_label):
                recs = by_label[label]
                rng.shuffle(recs)
                n = len(recs)
                n_tr = max(1, round(n * 0.5))
                n_va = max(1, round(n * 0.2))
                n_te = max(1, n - n_tr - n_va)
                if n_tr + n_va + n_te > n:
                    n_tr = n - n_va - n_te
                split["train"].extend(recs[:n_tr])
                split["val"].extend(recs[n_tr:n_tr + n_va])
                split["test"].extend(recs[n_tr + n_va:n_tr + n_va + n_te])

        write_json(split, str(split_path))
        # Marker so a later run recognizes the flat HF layout.
        try:
            (image_dir / ".hf_source").write_text(str(used))
        except OSError:
            pass
        print(f"[sun397] HF source '{used}': wrote {n_total} images and split "
              f"({len(split['train'])} train / {len(split['val'])} val / "
              f"{len(split['test'])} test) to {split_path}")

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
        if not image_dir.exists():
            raise RuntimeError(
                f"[sun397] cannot generate a split: image directory "
                f"'{image_dir}' does not exist. The SUN397 images must be "
                "downloaded/extracted first (see SUN397._download_images)."
            )

        print(f"[sun397] generating deterministic split (seed={seed})")

        # The canonical SUN397 tree is letter-prefixed (`a`, `b`, ...); the
        # HuggingFace fallback instead writes one flat `<NNN>/` folder per
        # class. Detect which layout we have.
        letter_layout = any(
            d.is_dir() and len(d.name) == 1 for d in image_dir.iterdir()
        )

        # Discover all class folders (deepest level that holds images).
        class_to_images = defaultdict(list)
        if letter_layout:
            for letter_dir in sorted(image_dir.iterdir()):
                if not letter_dir.is_dir() or len(letter_dir.name) != 1:
                    continue
                for scene_dir in sorted(letter_dir.iterdir()):
                    if not scene_dir.is_dir():
                        continue
                    # Some scenes nest one extra level (indoor/outdoor).
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
        else:
            # Flat layout: each top-level directory is a class.
            for cls_dir in sorted(image_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                rel_cls = cls_dir.relative_to(image_dir).as_posix()
                for img in sorted(cls_dir.rglob("*.jpg")):
                    class_to_images[rel_cls].append(img)

        def _classname(rel_cls: str) -> str:
            if letter_layout:
                # rel_cls looks like "a/airport_terminal" or "c/canyon/indoor".
                parts = rel_cls.split("/")[1:]   # drop leading letter
                parts = parts[::-1]              # indoor/outdoor comes first
                return " ".join(p.replace("_", " ") for p in parts)
            return rel_cls.split("/")[-1].replace("_", " ")

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
