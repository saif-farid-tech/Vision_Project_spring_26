# UCF101 Classification with SHViT

COE-486 Computer Vision — AUS Spring 2026.
UCF101 variant of the SHViT vs. baselines experiment. Data preprocessing
follows the [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)
project (`datasets/utils.py`): a CoOp-style train/val/test JSON split,
CLIP-style normalization, and bicubic resizing.

This branch reproduces the same pipeline as the original Food-101 experiment,
but using UCF101 (Soomro et al., 2012) — a 101-class human-action
benchmark. UCF101 is natively a video dataset; CoOp / Tip-Adapter recast it
as a still-image classification task by using the **mid-frame** of each
video clip as the input (~13,320 JPEGs, ~1.4 GB unpacked). Every notebook
writes its outputs under the root directory **`CV_Research_Paper_UCF101/`**
so the final artifacts mirror the stage layout one-to-one.

> **Data download note.** There is no public torchvision auto-download
> for the mid-frame JPGs — the canonical source is CoOp's shared Google
> Drive folder, fetched via `gdown`. `prepare_ucf101.py` attempts that
> automatically; if Drive rate-limits the download, manually unzip
> `UCF-101-midframes.zip` (and drop `split_zhou_UCF101.json` next to it)
> into `<data-root>/ucf101/` and re-run the prep script.
>
> The dataset is moderate-sized (~1.4 GB unpacked, ~130 images per class
> on average). Expect ~30 sec per epoch on a T4 (≈ 15 min per SHViT
> variant at 30 epochs). Mounting Drive is optional.

---

## Project description

We fine-tune all four SHViT variants (S1–S4) on UCF101 from
ImageNet-pretrained checkpoints using the paper's training recipe (cosine
LR with warmup, RandAugment, RandomErasing, Mixup / CutMix, label smoothing,
AGC-style gradient clipping). We train ResNet-50 and MobileNetV2 baselines
under the exact same data split and augmentation pipeline, and benchmark
every model end-to-end on top-1 / top-5 accuracy, parameter count, GFLOPs,
GPU throughput, and CPU latency.

Data preprocessing details:

- Split file: `<data-root>/ucf101/split_zhou_UCF101.json`
  (downloaded via `gdown` when available, otherwise generated locally as a
  deterministic per-class 50% / 20% / 30% train/val/test split — see
  `datasets/ucf101.py`).
- Image directory: `<data-root>/ucf101/UCF-101-midframes/<Action>/<image>.jpg`
  (one folder per action class, e.g. `Apply_Eye_Makeup/`,
  `Basketball_Dunk/`, `Pizza_Tossing/`, `Yo_Yo/`, …). The split JSON stores
  image paths relative to `<data-root>/ucf101/UCF-101-midframes/`. Folder
  names use underscore-separated capitalized tokens, matching the names
  CoOp / Tip-Adapter use as both the on-disk class folder and the
  classname stored in the split JSON.
- Download: handled by `prepare_ucf101.py`, which tries to fetch
  `UCF-101-midframes.zip` (and `split_zhou_UCF101.json`) from CoOp's
  shared Drive folder via `gdown`, then unpacks it under `<data-root>/ucf101/`.
- Normalization: CLIP statistics
  `mean=(0.48145466, 0.4578275, 0.40821073)`,
  `std=(0.26862954, 0.26130258, 0.27577711)`.
- Wrapping: `datasets/utils.py::DatasetWrapper` (bicubic resize, optional
  k-shot replication, optional return-of-img0 channel).

---

## Output root

**Every script and notebook in this branch writes its results under
`CV_Research_Paper_UCF101/`** so the final layout looks like:

```
CV_Research_Paper_UCF101/
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
- `prepare_ucf101.py` — fetches CoOp's pre-extracted UCF101 mid-frame
  archive (and the official split JSON) via `gdown`, unpacks into the
  Tip-Adapter folder convention. Falls back to a deterministic per-class
  50/20/30 split if the official split JSON is unavailable.
- `splits.py` — loads the Tip-Adapter UCF101 split into a
  torchvision-compatible Dataset (`load_split`, `load_test`, `load_all`).
- `augmentation.py` — training / validation transforms (CLIP normalization,
  bicubic resize, RandAugment + RandomErasing for train, Mixup helper).
- `metrics.py` — top-1 / top-5 accuracy, per-class accuracy, confusion matrix,
  precision / recall / F1, end-to-end `evaluate_model`.
- `eda.py` — UCF101 EDA visualizations (sample grid, image sizes,
  class distribution, summary text).
- `evaluate_all.py` — benchmarks all 6 trained models on the test split.
- `make_figures.py` — generates the headline figures from training logs +
  per-model `results.json`.
- `error_analysis.py` — detailed error analysis for SHViT-S4.
- `demo.py` — single-image inference with SHViT-S4 + label overlay.

**`datasets/`** — vendored from
[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets).
Contains the `Datum`, `DatasetBase`, `DatasetWrapper`, `UCF101`, and
`OxfordPets` helpers (the UCF101 loader reuses the static
`OxfordPets.read_split` helper for the CoOp JSON parsing), plus the
`auto_prepare` extension used by `prepare_ucf101.py`.

**Stage 1 — `Stage 1: testing the SHViT model/`**
- `setup_shvit.sh` — clones SHViT and installs its dependencies.
- `prepare_ucf101.py` — thin wrapper that calls the root prepare script.
- `verify_model.py` — sanity-check SHViT-S4 on a handful of UCF101 images.
- `SHViT_UCF101_Colab.ipynb` — Colab notebook for the stage.

**Stage 2 — `Stage 2: baseline models/`**
- `train_baseline.py` — fine-tunes ResNet-50 or MobileNetV2 on UCF101.
- `Baselines_UCF101_Colab.ipynb` — Colab notebook for the stage.

**Stage 3 — `Stage 3: fine-tuning SHViT/`**
- `finetune_shvit_ucf101.py` — fine-tunes any SHViT variant (S1..S4)
  on UCF101 with the paper's recipe.
- `SHViT_Finetune_UCF101_Colab.ipynb` — Colab notebook for the stage.

**Stage 4 — `Stage 4: Benchmarking and Demo/`**
- `Analysis_Pipeline_UCF101_Colab.ipynb` — ties `evaluate_all.py`,
  `make_figures.py`, `error_analysis.py`, and `demo.py` together.

---

## Quick local run

```bash
# 1. Prepare data
python prepare_ucf101.py --root data

# 2. Train baselines
python "Stage 2: baseline models/train_baseline.py" \
    --model resnet50 --data-root data \
    --output-dir "CV_Research_Paper_UCF101/Stage 2: baseline models"
python "Stage 2: baseline models/train_baseline.py" \
    --model mobilenet_v2 --data-root data \
    --output-dir "CV_Research_Paper_UCF101/Stage 2: baseline models"

# 3. Fine-tune SHViT variants (S1..S4) — requires SHViT repo + weights
python "Stage 3: fine-tuning SHViT/finetune_shvit_ucf101.py" \
    --model shvit_s4 --shvit-dir SHViT --finetune weights/shvit_s4.pth \
    --data-root data \
    --output-dir "CV_Research_Paper_UCF101/Stage 3: fine-tuning SHViT/shvit_s4"

# 4. Benchmark + figures + error analysis
python evaluate_all.py    --data-root data
python make_figures.py
python error_analysis.py  --data-root data
python demo.py --image sample.jpg --data-root data \
    --shvit-dir SHViT \
    --checkpoint "CV_Research_Paper_UCF101/Stage 3: fine-tuning SHViT/shvit_s4/best.pth"
```
