# Food Classification with SHViT — Tip-Adapter data pipeline variant

COE-486 Computer Vision — AUS Spring 2026.
Fine-tuning SHViT (Single-Head Vision Transformer, CVPR 2024) on Food-101 and comparing it against ResNet-50 and MobileNetV2 baselines.

> **This branch (`Vision_Project_spring_26_Food101`)** runs the *same* experiment as the main branch but swaps the **data preprocessing and data split** for the ones used by [Tip-Adapter's `datasets/utils.py`](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/utils.py):
> - **Data split:** the **Zhou split** (`split_zhou_Food101.json`, read via `OxfordPets.read_split`) instead of the previous deterministic 90/10 count-manifest split. The `food-101/images/<class>/*.jpg` layout is the one `torchvision.datasets.Food101` already produces; the split JSON is downloaded from Google Drive (file id `1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl`) into `<data_root>/food-101/`.
> - **Preprocessing:** Tip-Adapter's `DatasetWrapper` transform — **BICUBIC resize → ToTensor → CLIP normalization** (mean `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)`) — for **both** train and val/test, with **no** RandAugment / RandomErasing / Mixup / CutMix. Training therefore uses plain cross-entropy on hard labels.
> - **Output location:** every Colab notebook saves all of its files (dataset, weights, checkpoints, analysis artifacts, bundles) under a single root directory called **`CV_Research_Paper_Food101`**.
>
> The ported pipeline lives in the `tip_datasets/` package; `splits.py` and `augmentation.py` are thin wrappers over it.

---

## Project description

This project investigates whether SHViT — a lightweight transformer designed for mobile and edge inference — can match the accuracy of established CNN baselines (ResNet-50, MobileNetV2) on the 101-class Food-101 fine-grained food recognition benchmark, while remaining cheap enough to run interactively. We fine-tune all four SHViT variants (S1–S4) from ImageNet-pretrained checkpoints (cosine LR with warmup, gradient clipping), train both CNN baselines under the exact same data split and preprocessing so the comparison is apples-to-apples, and benchmark every model end-to-end on accuracy, parameter count, GFLOPs, GPU throughput, and CPU latency. On this branch the data split and preprocessing follow Tip-Adapter (Zhou split + CLIP-normalized BICUBIC resize, no augmentation), as described in the note above.

> **Note on committed artifacts:** the checkpoints, training logs, `results.json` dumps, and figures currently committed under Stages 2–4 were produced on the *main* branch (90/10 count-manifest split + ImageNet/RandAugment preprocessing). They are carried over so the repo layout is complete; re-run the notebooks to regenerate them under the Tip-Adapter pipeline — the headline numbers below will change accordingly.

## What we achieved

We trained and benchmarked all six models on a shared 90/10 train/val split (seed 42) and evaluated on the official Food-101 test set, finishing with ResNet-50 at 89.74% top-1, SHViT-S4 at 85.67%, MobileNetV2 at 85.20%, SHViT-S3 at 84.97%, SHViT-S2 at 82.40%, and SHViT-S1 at 80.04% — placing SHViT-S4 within ~4 points of ResNet-50 while using roughly 1/5 the GFLOPs (0.79 vs. 4.11) and beating ResNet-50 on GPU throughput. We additionally produced a full reproducibility package: deterministic split manifests for three seeds, per-epoch training logs and best checkpoints for every model, a unified evaluation script that emits JSON results plus Markdown/LaTeX summary tables, headline figures (training curves, train-loss curves, accuracy bars, speed-vs-accuracy scatter), an error analysis of SHViT-S4's worst classes and most-confused pairs, and a single-image demo that runs SHViT-S4 inference end-to-end.

---

## Repo files

**Root files**
- `README.md` — this document; orients a new reader to the repo layout, the project goal, and the headline results.
- `.gitignore` — keeps the downloaded Food-101 dataset, local checkpoint caches, and Python/Colab build artifacts out of version control.
- `augmentation.py` — on this branch, returns Tip-Adapter's preprocessing transform (BICUBIC resize + ToTensor + CLIP normalization) for both the train and val transforms; `build_mixup_fn` returns `None` (no Mixup/CutMix). No RandAugment/RandomErasing.
- `metrics.py` — provides top-1/top-5 accuracy, per-class accuracy, confusion matrix, and precision/recall/F1 used to evaluate every model.
- `splits.py` — thin wrapper over `tip_datasets`: builds the Tip-Adapter `Food101` object (Zhou split) and returns train/val/test `DatasetWrapper`s plus the label-ordered class names.
- `tip_datasets/` — ported Tip-Adapter data package: `utils.py` (`Datum`/`DatasetBase`/`DatasetWrapper`/`build_data_loader` + CLIP preprocessing), `oxford_pets.py` (`read_split`/`save_split`), and `food101.py` (the Food-101 Zhou-split dataset class).
- `train_val_split_seed{42,123,456}.json` — the previous 90/10 count-manifest splits. **Unused on this branch** (kept for reference); the split now comes from `split_zhou_Food101.json`.
- `eda.py` — generates the report's exploratory data analysis (sample grid, image-size histograms, class-distribution bars, summary text) for the Food-101 dataset.
- `evaluate_all.py` — runs inference and benchmarks (params, GFLOPs, GPU throughput, CPU latency, top-1/top-5, confusion matrix) for all six trained models and writes per-model JSON plus Markdown/LaTeX summary tables.
- `make_figures.py` — builds the headline figures (validation-accuracy curves, train-loss curves, accuracy bar chart, speed-vs-accuracy scatter) from the training logs and `evaluate_all.py` outputs.
- `error_analysis.py` — drills into SHViT-S4's `results.json` to produce the worst-classes bar chart, top-20 confusion heatmap, misclassified-image grid, and a summary of best/worst classes and most-confused pairs.
- `demo.py` — runs single-image inference with the fine-tuned SHViT-S4 checkpoint, prints top-5 predictions, and saves the input image with the predicted label overlaid.

**Stage 1 — Testing the SHViT model** (`Stage 1: testing the SHViT model/`)
- `setup_shvit.sh` — clones the SHViT repo and installs its dependencies in one command for local setup.
- `prepare_food101.py` — downloads Food-101 and arranges it into the folder structure expected by the training scripts.
- `verify_model.py` — loads the SHViT-S4 ImageNet checkpoint and runs it on a handful of Food-101 images to confirm the model imports and runs without errors.
- `SHViT_Food101_Colab.ipynb` — Colab notebook that reproduces Stage 1 end-to-end on a free T4 GPU.

**Stage 2 — Baseline models** (`Stage 2: baseline models/`)
- `train_baseline.py` — fine-tunes ResNet-50 or MobileNetV2 on Food-101 using the shared split and augmentation modules.
- `Baselines_Food101_Colab.ipynb` — Colab notebook that runs both baseline models and plots the training curves.
- `README.md` — short note recording the best validation top-1 reached by each baseline.
- `resnet50/best.pth` — saved checkpoint of the best ResNet-50 weights (val top-1 = 85.70%).
- `resnet50/training_log.csv` — per-epoch loss and accuracy log from the completed ResNet-50 training run.
- `resnet50/README.md` — note describing the ResNet-50 checkpoint and its best validation top-1.
- `mobilenet_v2/best.pth` — saved checkpoint of the best MobileNetV2 weights (val top-1 = 80.82%).
- `mobilenet_v2/training_log.csv` — per-epoch loss and accuracy log from the completed MobileNetV2 training run.
- `mobilenet_v2/README.md` — note describing the MobileNetV2 checkpoint and its best validation top-1.

**Stage 3 — Fine-tuning SHViT** (`Stage 3: fine-tuning SHViT/`)
- `finetune_shvit_food101.py` — fine-tunes any SHViT variant (S1–S4) on Food-101 with the paper's recipe (cosine LR, warmup, Mixup, gradient clipping) using the shared split and augmentation modules.
- `SHViT_Finetune_Colab.ipynb` — Colab notebook that runs the full 30-epoch SHViT fine-tuning and plots training curves.
- `README.md` — table of best validation top-1/top-5 per SHViT variant alongside a training-curves figure.
- `shvit_s1/best.pth` — best SHViT-S1 checkpoint (val top-1 = 74.51%).
- `shvit_s1/training_log.csv` — per-epoch SHViT-S1 training log.
- `shvit_s1/README.md` — note recording SHViT-S1's evaluation top-1/top-5 and a Drive link to intermediate checkpoints.
- `shvit_s2/best.pth` — best SHViT-S2 checkpoint (val top-1 = 77.58%).
- `shvit_s2/training_log.csv` — per-epoch SHViT-S2 training log.
- `shvit_s2/README.md` — note recording SHViT-S2's evaluation top-1/top-5 and a Drive link to intermediate checkpoints.
- `shvit_s3/best.pth` — best SHViT-S3 checkpoint (val top-1 = 80.16%).
- `shvit_s3/training_log.csv` — per-epoch SHViT-S3 training log.
- `shvit_s3/README.md` — note recording SHViT-S3's evaluation top-1/top-5 and a Drive link to intermediate checkpoints.
- `shvit_s4/best.pth` — best SHViT-S4 checkpoint (val top-1 = 80.73%), used for the demo and error analysis.
- `shvit_s4/training_log.csv` — per-epoch SHViT-S4 training log.
- `shvit_s4/README.md` — note recording SHViT-S4's evaluation top-1/top-5 and a Drive link to intermediate checkpoints.

**Stage 4 — Benchmarking and Demo** (`Stage 4: Benchmarking and Demo/`)
- `Analysis_Pipeline_Colab.ipynb` — Colab notebook that ties the analysis pipeline together: runs `evaluate_all.py`, `make_figures.py`, `error_analysis.py`, and the SHViT-S4 demo.
- `README.md` — pointer note explaining that this stage is where all six models are benchmarked and SHViT-S4 is demoed.
- `analysis/demo_output/demo_output.png` — sample output of `demo.py` (a Food-101 image overlaid with SHViT-S4's top prediction and confidence).
- `analysis/eda_outputs/eda_samples.png` — 5×5 grid of random Food-101 training images with class labels.
- `analysis/eda_outputs/eda_image_sizes.png` — width/height histograms over a random sample of training images.
- `analysis/eda_outputs/eda_class_distribution.png` — stacked bar chart of per-class image counts across train and test splits.
- `analysis/eda_outputs/eda_summary.txt` — text summary of dataset statistics (totals, per-class counts, image-size stats, class list).
- `analysis/error_analysis_outputs/confusion_top20.png` — row-normalized confusion heatmap restricted to SHViT-S4's 20 worst classes.
- `analysis/error_analysis_outputs/worst_15_categories.png` — horizontal bar chart of SHViT-S4's 15 lowest per-class accuracies.
- `analysis/error_analysis_outputs/misclassified_grid.png` — 4×4 grid of randomly sampled SHViT-S4 misclassifications with true/predicted labels and confidences.
- `analysis/error_analysis_outputs/error_analysis_summary.txt` — text summary listing SHViT-S4's worst-10 and best-10 classes plus its top-10 most confused class pairs.
- `analysis/figures/fig_training_curves.png` — validation top-1 accuracy over epochs for all six models.
- `analysis/figures/fig_train_loss.png` — training loss over epochs for all six models.
- `analysis/figures/fig_accuracy_bars.png` — sorted bar chart of test top-1 accuracy across all six models.
- `analysis/figures/fig_speed_vs_accuracy.png` — scatter of GPU throughput against test top-1 accuracy for all six models.
- `analysis/results/results_table.md` — Markdown summary table (params, GFLOPs, top-1/top-5, GPU throughput, CPU latency) for all six models.
- `analysis/results/results_table.tex` — LaTeX version of the same summary table for the report.
- `analysis/results/<model>/results.json` — per-model evaluation dump (top-1/top-5, per-class accuracy, full confusion matrix, all preds/targets, params, GFLOPs, GPU throughput, CPU latency) for `resnet50`, `mobilenet_v2`, and `shvit_s1`–`shvit_s4`.

---

## Repo directories

- `Stage 1: testing the SHViT model/` — scripts and a Colab notebook that bring up the SHViT codebase, prepare Food-101, and sanity-check the ImageNet-pretrained SHViT-S4 model before any fine-tuning.
- `Stage 2: baseline models/` — training script, Colab notebook, checkpoints, and per-epoch logs for the ResNet-50 and MobileNetV2 baselines used to compare against SHViT.
- `Stage 2: baseline models/resnet50/` — saved best checkpoint, training log, and short note for the fine-tuned ResNet-50 baseline.
- `Stage 2: baseline models/mobilenet_v2/` — saved best checkpoint, training log, and short note for the fine-tuned MobileNetV2 baseline.
- `Stage 3: fine-tuning SHViT/` — fine-tuning script, Colab notebook, and per-variant checkpoints/logs for all four SHViT variants on Food-101.
- `Stage 3: fine-tuning SHViT/shvit_s1/` — best checkpoint, training log, and note for the fine-tuned SHViT-S1 model.
- `Stage 3: fine-tuning SHViT/shvit_s2/` — best checkpoint, training log, and note for the fine-tuned SHViT-S2 model.
- `Stage 3: fine-tuning SHViT/shvit_s3/` — best checkpoint, training log, and note for the fine-tuned SHViT-S3 model.
- `Stage 3: fine-tuning SHViT/shvit_s4/` — best checkpoint, training log, and note for the fine-tuned SHViT-S4 model used in the demo and error analysis.
- `Stage 4: Benchmarking and Demo/` — Colab notebook and analysis artifacts that benchmark all six trained models and demo SHViT-S4.
- `Stage 4: Benchmarking and Demo/analysis/` — home of every generated artifact in Stage 4 (EDA, error analysis, figures, demo image, per-model results).
- `Stage 4: Benchmarking and Demo/analysis/demo_output/` — saved output image from `demo.py` showing a Food-101 photo with SHViT-S4's top prediction overlaid.
- `Stage 4: Benchmarking and Demo/analysis/eda_outputs/` — exploratory data analysis figures and summary text generated by `eda.py`.
- `Stage 4: Benchmarking and Demo/analysis/error_analysis_outputs/` — confusion heatmaps, worst-classes chart, misclassified-image grid, and summary text generated by `error_analysis.py` for SHViT-S4.
- `Stage 4: Benchmarking and Demo/analysis/figures/` — headline figures (training curves, train-loss, accuracy bars, speed-vs-accuracy scatter) produced by `make_figures.py`.
- `Stage 4: Benchmarking and Demo/analysis/results/` — Markdown/LaTeX summary tables and per-model `results.json` dumps produced by `evaluate_all.py`.
- `Stage 4: Benchmarking and Demo/analysis/results/resnet50/` — `results.json` from `evaluate_all.py` for ResNet-50.
- `Stage 4: Benchmarking and Demo/analysis/results/mobilenet_v2/` — `results.json` from `evaluate_all.py` for MobileNetV2.
- `Stage 4: Benchmarking and Demo/analysis/results/shvit_s1/` — `results.json` from `evaluate_all.py` for SHViT-S1.
- `Stage 4: Benchmarking and Demo/analysis/results/shvit_s2/` — `results.json` from `evaluate_all.py` for SHViT-S2.
- `Stage 4: Benchmarking and Demo/analysis/results/shvit_s3/` — `results.json` from `evaluate_all.py` for SHViT-S3.
- `Stage 4: Benchmarking and Demo/analysis/results/shvit_s4/` — `results.json` from `evaluate_all.py` for SHViT-S4.
