"""
train_baseline.py
Fine-tune ResNet-50 or MobileNetV2 on FGVC Aircraft as a baseline for SHViT.

Splits: FGVC Aircraft train / val / test as defined by the Tip-Adapter style
        FGVC Aircraft's canonical split files (provided by
        prepare_fgvc_aircraft.py).  The test split is reserved for Stage 4's
        evaluate_all.py and never touched here.

Augmentation: shared with Stage 3 via augmentation.py at the repo root
              (RandAugment + RandomErasing for train, Resize+CenterCrop for val).

Usage:
    python train_baseline.py --model resnet50      --data-root data \
        --output-dir "CV_Research_Paper_FGVCAircraft/Stage 2: baseline models"
    python train_baseline.py --model mobilenet_v2  --data-root data \
        --output-dir "CV_Research_Paper_FGVCAircraft/Stage 2: baseline models"

Outputs (per model):
    <output-dir>/<model>/training_log.csv   one row per epoch
    <output-dir>/<model>/best.pth           checkpoint of highest val top-1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
)

# metrics.py, splits.py, augmentation.py live at the repo root.
# Insert both the script's own directory (for Colab flat copies) and its
# parent (for local runs from within the repo tree).
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [str(_SCRIPT_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import top_k_accuracy                                                   # noqa: E402
from augmentation import build_train_transform, build_val_transform, build_mixup_fn  # noqa: E402
import splits                                                                         # noqa: E402


NUM_CLASSES = 100  # FGVC Aircraft under the Tip-Adapter / CoOp split


def build_dataloaders(args):
    splits.ensure_prepared(args.data_root)
    train_ds, val_ds = splits.load_split(
        args.data_root,
        train_transform=build_train_transform(),
        val_transform=build_val_transform(),
    )
    print(f"FGVC Aircraft split: {len(train_ds):,} train  {len(val_ds):,} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_model(name: str) -> nn.Module:
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_fn, clip_grad):
    """Matches the SHViT recipe in finetune_shvit_fgvc_aircraft.py."""
    model.train()
    total_loss, n = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_outputs, all_targets_list = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_outputs.append(logits.cpu())
        all_targets_list.append(targets.cpu())
        n += images.size(0)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)
    top1 = top_k_accuracy(all_outputs, all_targets, k=1) / 100.0
    top5 = top_k_accuracy(all_outputs, all_targets, k=5) / 100.0
    return total_loss / n, top1, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["resnet50", "mobilenet_v2"], required=True)
    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--output-dir",
                        default="CV_Research_Paper_FGVCAircraft/Stage 2: baseline models")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int,   default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}   Device: {device}   Epochs: {args.epochs}")

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_log.csv"
    best_path = out_dir / "best.pth"

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    model = build_model(args.model).to(device)

    mixup_fn = build_mixup_fn(num_classes=NUM_CLASSES)
    criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.cuda.amp.GradScaler()
    clip_grad = 0.02

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss", "val_top1", "val_top5", "time_sec"]
        )

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, mixup_fn, clip_grad,
        )
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"lr={lr:.2e}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  top1={val_top1*100:.2f}%  "
            f"top5={val_top5*100:.2f}%  ({elapsed:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.6e}",
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{val_top1:.4f}", f"{val_top5:.4f}",
                f"{elapsed:.1f}",
            ])

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({
                "model_name": args.model,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
            }, best_path)
            print(f"  -> new best, saved to {best_path}")

    print(f"\nDone. Best val top-1: {best_top1*100:.2f}%")
    print(f"Log:  {csv_path}")
    print(f"Ckpt: {best_path}")


if __name__ == "__main__":
    main()
