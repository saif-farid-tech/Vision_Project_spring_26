# FGVC Aircraft Classification with SHViT

COE-486 Computer Vision — AUS Spring 2026.
FGVC Aircraft variant of the SHViT vs. baselines experiment. Data
preprocessing follows the
[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) project
(`datasets/utils.py`): CLIP-style normalization, bicubic resizing, and the
Tip-Adapter `DatasetWrapper` (unlike the other dataset variants in this
project, FGVC Aircraft uses its own canonical train/val/test split files,
so no CoOp JSON split is involved).

This branch reproduces the same pipeline as the original Food-101 experiment,
but using FGVC Aircraft (100 fine-grained aircraft variants from Maji et al.,
2013). Every notebook writes its outputs under the root directory
**`CV_Research_Paper_FGVCAircraft/`** so the final artifacts mirror the stage
layout one-to-one.

---

## Project description

We fine-tune all four SHViT variants (S1–S4) on FGVC Aircraft from
ImageNet-pretrained checkpoints using the paper's training recipe (cosine
LR with warmup, RandAugment, RandomErasing, Mixup / CutMix, label smoothing,
AGC-style gradient clipping). We train ResNet-50 and MobileNetV2 baselines
under the exact same data split and augmentation pipeline, and benchmark
every model end-to-end on top-1 / top-5 accuracy, parameter count, GFLOPs,
GPU throughput, and CPU latency.

Data preprocessing details:

- Split files: `<data-root>/fgvc_aircraft/images_variant_train.txt`,
  `images_variant_val.txt`, and `images_variant_test.txt` — the canonical
  FGVC Aircraft splits (no CoOp JSON, no random partitioning).
- Class list: `<data-root>/fgvc_aircraft/variants.txt` (one of 100
  aircraft variants per line, e.g. `707-320`, `Boeing 707`, `Cessna 172`).
- Image directory: `<data-root>/fgvc_aircraft/images/<id>.jpg` — flat
  directory of JPEGs named by numeric image id; the variant for each id
  is recorded in the split files.
- Download: `torchvision.datasets.FGVCAircraft` fetches and extracts the
  upstream tar from Maji et al.; the prep script re-shapes the unpack
  (`fgvc-aircraft-2013b/data/{...}`) into the Tip-Adapter style layout
  above.
- Normalization: CLIP statistics
  `mean=(0.48145466, 0.4578275, 0.40821073)`,
  `std=(0.26862954, 0.26130258, 0.27577711)`.
- Wrapping: `datasets/utils.py::DatasetWrapper` (bicubic resize, optional
  k-shot replication, optional return-of-img0 channel).

---

## Output root

**Every script and notebook in this branch writes its results under
`CV_Research_Paper_FGVCAircraft/`** so the final layout looks like:

```
CV_Research_Paper_FGVCAircraft/
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
- `prepare_fgvc_aircraft.py` — downloads FGVC Aircraft via torchvision
  and re-layouts it into the Tip-Adapter style folder convention.
- `splits.py` — loads the Tip-Adapter FGVC Aircraft split into a
  torchvision-compatible Dataset (`load_split`, `load_test`, `load_all`).
- `augmentation.py` — training / validation transforms (CLIP normalization,
  bicubic resize, RandAugment + RandomErasing for train, Mixup helper).
- `metrics.py` — top-1 / top-5 accuracy, per-class accuracy, confusion matrix,
  precision / recall / F1, end-to-end `evaluate_model`.
- `eda.py` — FGVC Aircraft EDA visualizations (sample grid, image sizes,
  class distribution, summary text).
- `evaluate_all.py` — benchmarks all 6 trained models on the test split.
- `make_figures.py` — generates the headline figures from training logs +
  per-model `results.json`.
- `error_analysis.py` — detailed error analysis for SHViT-S4.
- `demo.py` — single-image inference with SHViT-S4 + label overlay.

**`datasets/`** — vendored from
[Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/tree/main/datasets).
Contains the `Datum`, `DatasetBase`, `DatasetWrapper`, and `FGVCAircraft`
helpers, plus the `auto_prepare` extension used by `prepare_fgvc_aircraft.py`.

**Stage 1 — `Stage 1: testing the SHViT model/`**
- `setup_shvit.sh` — clones SHViT and installs its dependencies.
- `prepare_fgvc_aircraft.py` — thin wrapper that calls the root prepare script.
- `verify_model.py` — sanity-check SHViT-S4 on a handful of FGVC Aircraft images.
- `SHViT_FGVCAircraft_Colab.ipynb` — Colab notebook for the stage.

**Stage 2 — `Stage 2: baseline models/`**
- `train_baseline.py` — fine-tunes ResNet-50 or MobileNetV2 on FGVC Aircraft.
- `Baselines_FGVCAircraft_Colab.ipynb` — Colab notebook for the stage.

**Stage 3 — `Stage 3: fine-tuning SHViT/`**
- `finetune_shvit_fgvc_aircraft.py` — fine-tunes any SHViT variant (S1..S4)
  on FGVC Aircraft with the paper's recipe.
- `SHViT_Finetune_FGVCAircraft_Colab.ipynb` — Colab notebook for the stage.

**Stage 4 — `Stage 4: Benchmarking and Demo/`**
- `Analysis_Pipeline_FGVCAircraft_Colab.ipynb` — ties `evaluate_all.py`,
  `make_figures.py`, `error_analysis.py`, and `demo.py` together.

---

## Quick local run

```bash
# 1. Prepare data
python prepare_fgvc_aircraft.py --root data

# 2. Train baselines
python "Stage 2: baseline models/train_baseline.py" \
    --model resnet50 --data-root data \
    --output-dir "CV_Research_Paper_FGVCAircraft/Stage 2: baseline models"
python "Stage 2: baseline models/train_baseline.py" \
    --model mobilenet_v2 --data-root data \
    --output-dir "CV_Research_Paper_FGVCAircraft/Stage 2: baseline models"

# 3. Fine-tune SHViT variants (S1..S4) — requires SHViT repo + weights
python "Stage 3: fine-tuning SHViT/finetune_shvit_fgvc_aircraft.py" \
    --model shvit_s4 --shvit-dir SHViT --finetune weights/shvit_s4.pth \
    --data-root data \
    --output-dir "CV_Research_Paper_FGVCAircraft/Stage 3: fine-tuning SHViT/shvit_s4"

# 4. Benchmark + figures + error analysis
python evaluate_all.py    --data-root data
python make_figures.py
python error_analysis.py  --data-root data
python demo.py --image sample.jpg --data-root data \
    --shvit-dir SHViT \
    --checkpoint "CV_Research_Paper_FGVCAircraft/Stage 3: fine-tuning SHViT/shvit_s4/best.pth"
```
