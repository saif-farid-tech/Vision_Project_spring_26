# SUN397 Classification with SHViT

COE-486 Computer Vision — AUS Spring 2026.
SUN397 variant of the SHViT vs. baselines experiment. Data preprocessing
follows the [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)
project (`datasets/utils.py`): a CoOp-style train/val/test JSON split,
CLIP-style normalization, and bicubic resizing.

This branch reproduces the same pipeline as the original Food-101 experiment,
but using SUN397 (Xiao et al., 2010) — 397 fine-grained scene categories
across ~108k images. Every notebook writes its outputs under the root
directory **`CV_Research_Paper_SUN397/`** so the final artifacts mirror the
stage layout one-to-one.

> **Heads up:** SUN397 is large (~37 GB unpacked). Mount Google Drive in
> the Colab notebooks (`USE_DRIVE = True`) so the dataset and checkpoints
> survive between sessions. Each fine-tuning epoch takes considerably
> longer than for the smaller datasets in the sister branches.

---

## Project description

We fine-tune all four SHViT variants (S1–S4) on SUN397 from
ImageNet-pretrained checkpoints using the paper's training recipe (cosine
LR with warmup, RandAugment, RandomErasing, Mixup / CutMix, label smoothing,
AGC-style gradient clipping). We train ResNet-50 and MobileNetV2 baselines
under the exact same data split and augmentation pipeline, and benchmark
every model end-to-end on top-1 / top-5 accuracy, parameter count, GFLOPs,
GPU throughput, and CPU latency.

Data preprocessing details:

- Split file: `<data-root>/sun397/split_zhou_SUN397.json`
  (downloaded via `gdown` when available, otherwise generated locally as a
  deterministic per-class 50% / 20% / 30% train/val/test split — see
  `datasets/sun397.py`).
- Image directory: `<data-root>/sun397/SUN397/<letter>/<scene>/<image>.jpg`
  (the canonical SUN397 layout has a letter-prefixed first level — `a`, `b`,
  …, `y` — with scene-name subfolders and, for some scenes, an extra
  `indoor/outdoor` nesting). The split JSON stores image paths relative to
  `<data-root>/sun397/SUN397/`.
- Download: the prep script fetches and extracts the upstream SUN397 archive
  (HTTPS Princeton URL first, with the plain-HTTP URL as a fallback), then
  re-shapes the unpack into the Tip-Adapter style layout above. If the
  official host is unreachable, point it at a reachable mirror with
  `SUN397_URL=<url> python prepare_sun397.py --root data`, or download
  `SUN397.tar.gz` manually and extract it so that
  `<data-root>/sun397/SUN397/<letter>/<scene>/<image>.jpg` exists before
  re-running. A checksum mismatch on the upstream tarball is tolerated
  (the download is retried without MD5 verification).
- Normalization: CLIP statistics
  `mean=(0.48145466, 0.4578275, 0.40821073)`,
  `std=(0.26862954, 0.26130258, 0.27577711)`.
- Wrapping: `datasets/utils.py::DatasetWrapper` (bicubic resize, optional
  k-shot replication, optional return-of-img0 channel).

---

## Output root

**Every script and notebook in this branch writes its results under
`CV_Research_Paper_SUN397/`** so the final layout looks like:

```
CV_Research_Paper_SUN397/
├── Stage 2: baseline models/
│   ├── resnet50/{training_log.csv, best.pth}
│   └── mobilenet_v2/{training_log.csv, best.pth}
├── Stage 3: fine-tuning SHViT/
│   ├── shvit_s1/{training_log.csv, best.pth, checkpoint_*.pth}
│   ├── shvit_s2/…
│   ├── shvit_s3/…
│   └── shvit_s4/…
└── Stage 4: Benchmarking and Demo/
    └── analysis/
        ├── eda_outputs/             (eda.py)
        ├── results/                 (evaluate_all.py)
        ├── figures/                 (make_figures.py)
        ├── error_analysis_outputs/  (error_analysis.py)
        └── demo_output/             (demo.py)
```

---

## Repo layout

**Root**
- `README.md` — this file.
- `prepare_sun397.py` — downloads SUN397 via torchvision, re-layouts the
  image tree into the Tip-Adapter folder convention, and prepares the
  CoOp split JSON (gdown-first, deterministic fallback).
- `splits.py` — loads the Tip-Adapter SUN397 split into a
  torchvision-compatible Dataset (`load_split`, `load_test`, `load_all`).
- `augmentation.py` — training / validation transforms (CLIP normalization,
  bicubic resize, RandAugment + RandomErasing for train, Mixup helper).
- `metrics.py` — top-1 / top-5 accuracy, per-class accuracy, confusion matrix,
  precision / recall / F1, end-to-end `evaluate_model`.
- `eda.py` — SUN397 EDA visualizations (sample grid, image sizes,
  class distribution, summary text).
- `evaluate_all.py` — benchmarks all 6 trained models on the test split.
- `make_figures.py` — generates the headline figures from training logs +
  per-model `results.json`.
- `error_analysis.py` — detailed error analysis for SHViT-S4.
- `demo.py` — single-image inference with SHViT-S4 + label overlay.

**`datasets/`** — vendored from
[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets).
Contains the `Datum`, `DatasetBase`, `DatasetWrapper`, `SUN397`, and
`OxfordPets` helpers (the SUN397 loader reuses the static
`OxfordPets.read_split` helper for the CoOp JSON parsing), plus the
`auto_prepare` extension used by `prepare_sun397.py`.

**Stage 1 — `Stage 1: testing the SHViT model/`**
- `setup_shvit.sh` — clones SHViT and installs its dependencies.
- `prepare_sun397.py` — thin wrapper that calls the root prepare script.
- `verify_model.py` — sanity-check SHViT-S4 on a handful of SUN397 images.
- `SHViT_SUN397_Colab.ipynb` — Colab notebook for the stage.

**Stage 2 — `Stage 2: baseline models/`**
- `train_baseline.py` — fine-tunes ResNet-50 or MobileNetV2 on SUN397.
- `Baselines_SUN397_Colab.ipynb` — Colab notebook for the stage.

**Stage 3 — `Stage 3: fine-tuning SHViT/`**
- `finetune_shvit_sun397.py` — fine-tunes any SHViT variant (S1..S4)
  on SUN397 with the paper's recipe.
- `SHViT_Finetune_SUN397_Colab.ipynb` — Colab notebook for the stage.

**Stage 4 — `Stage 4: Benchmarking and Demo/`**
- `Analysis_Pipeline_SUN397_Colab.ipynb` — ties `evaluate_all.py`,
  `make_figures.py`, `error_analysis.py`, and `demo.py` together.

---

## Quick local run

```bash
# 1. Prepare data
python prepare_sun397.py --root data

# 2. Train baselines
python "Stage 2: baseline models/train_baseline.py" \
    --model resnet50 --data-root data \
    --output-dir "CV_Research_Paper_SUN397/Stage 2: baseline models"
python "Stage 2: baseline models/train_baseline.py" \
    --model mobilenet_v2 --data-root data \
    --output-dir "CV_Research_Paper_SUN397/Stage 2: baseline models"

# 3. Fine-tune SHViT variants (S1..S4) — requires SHViT repo + weights
python "Stage 3: fine-tuning SHViT/finetune_shvit_sun397.py" \
    --model shvit_s4 --shvit-dir SHViT --finetune weights/shvit_s4.pth \
    --data-root data \
    --output-dir "CV_Research_Paper_SUN397/Stage 3: fine-tuning SHViT/shvit_s4"

# 4. Benchmark + figures + error analysis
python evaluate_all.py    --data-root data
python make_figures.py
python error_analysis.py  --data-root data
python demo.py --image sample.jpg --data-root data \
    --shvit-dir SHViT \
    --checkpoint "CV_Research_Paper_SUN397/Stage 3: fine-tuning SHViT/shvit_s4/best.pth"
```
