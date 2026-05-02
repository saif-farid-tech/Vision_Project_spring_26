# Food Classification with SHViT

COE-486 Computer Vision — AUS Spring 2026.
Fine-tuning SHViT (Single-Head Vision Transformer, CVPR 2024) on Food-101 and comparing it against ResNet-50 and MobileNetV2 baselines.

---

## Repo files

**Shared modules (repo root)**
- `augmentation.py` — defines the training and validation transforms (RandAugment + RandomErasing) and the Mixup/CutMix function used by both baselines and SHViT.
- `metrics.py` — provides top-1/top-5 accuracy, per-class accuracy, confusion matrix, and precision/recall/F1 used to evaluate every model.
- `splits.py` — converts Ahmed's count-manifest JSONs into deterministic train/val `torch.Subset` pairs so every model is evaluated on the same held-out images.
- `train_val_split_seed42.json` — the primary 90/10 train/val split (seed 42) used in all training runs.
- `train_val_split_seed123.json` — alternative split with seed 123 for reproducibility checks.
- `train_val_split_seed456.json` — alternative split with seed 456 for reproducibility checks.

**Stage 1 — Testing the SHViT model** (`Stage 1: testing the SHViT model/`)
- `setup_shvit.sh` — clones the SHViT repo and installs its dependencies in one command for local setup.
- `prepare_food101.py` — downloads Food-101 and arranges it into the folder structure expected by the training scripts.
- `verify_model.py` — loads the SHViT-S4 ImageNet checkpoint and runs it on a handful of Food-101 images to confirm the model imports and runs without errors.
- `SHViT_Food101_Colab.ipynb` — Colab notebook that reproduces Stage 1 end-to-end on a free T4 GPU.

**Stage 2 — Baseline models** (`Stage 2: baseline models/`)
- `train_baseline.py` — fine-tunes ResNet-50 or MobileNetV2 on Food-101 using the shared split and augmentation modules.
- `Baselines_Food101_Colab.ipynb` — Colab notebook that runs both baseline models and plots the training curves.
- `resnet50/training_log.csv` — per-epoch loss and accuracy log from the completed ResNet-50 training run.
- `mobilenet_v2/training_log.csv` — per-epoch loss and accuracy log from the completed MobileNetV2 training run.
- `mobilenet_v2/best.pth` — saved checkpoint of the best MobileNetV2 weights.
- `base_models_over epochs.png` — validation accuracy curves comparing both baselines across epochs.
- `best_val_top-1.rtf` — quick summary of the best top-1 scores reached by each baseline.

**Stage 3 — Fine-tuning SHViT** (`Stage 3: fine-tuning SHViT/`)
- `finetune_shvit_food101.py` — fine-tunes SHViT-S4 on Food-101 with the paper's recipe (cosine LR, warmup, Mixup, gradient clipping) using the shared split and augmentation modules.
- `SHViT_Finetune_Colab.ipynb` — Colab notebook that runs the full 30-epoch SHViT fine-tuning and plots training curves.
