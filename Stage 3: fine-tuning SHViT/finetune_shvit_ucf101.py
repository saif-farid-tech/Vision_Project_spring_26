"""
finetune_shvit_ucf101.py

Single-GPU wrapper for fine-tuning SHViT (S1..S4) on UCF101.

Design choices that match the original SHViT paper / engine.py:
  - Augmentation: defined in augmentation.py at the repo root
    (RandAugment(2, 9) + RandomErasing(p=0.25) train, plain resize+crop val,
    Mixup(0.8) + CutMix(1.0) + label smoothing 0.1).
  - Forward pass during training runs in full FP32 (matching the commented-out
    autocast in the original engine.py's train_one_epoch).
  - Eval forward pass uses torch.cuda.amp.autocast (matching original evaluate()).
  - GradScaler is kept for gradient overflow protection.
  - Gradient clipping via AGC-style norm clip (clip_grad=0.02, matching paper).
  - Head keys dropped by shape mismatch (same logic as main.py --finetune path).
  - LR is used directly — no linear world-size scaling since world_size=1.

Usage:
    python finetune_shvit_ucf101.py \\
        --shvit-dir  /path/to/SHViT \\
        --finetune   /path/to/shvit_s4.pth \\
        --data-root  /path/to/data \\
        --output-dir "CV_Research_Paper_UCF101/Stage 3: fine-tuning SHViT/shvit_s4" \\
        --epochs 30

Outputs:
    <output-dir>/training_log.csv        one row per epoch
    <output-dir>/best.pth                highest val top-1 (model state_dict only)
    <output-dir>/checkpoint_<N>.pth      resumable checkpoint every --save-freq epochs
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import SoftTargetCrossEntropy
from timm.models import create_model

# metrics.py, splits.py, augmentation.py live at the repo root.
_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [str(_SCRIPT_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import top_k_accuracy                                                   # noqa: E402
from augmentation import build_train_transform, build_val_transform, build_mixup_fn  # noqa: E402
import splits                                                                         # noqa: E402


NUM_CLASSES = 101  # UCF101 under the Tip-Adapter / CoOp split


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("SHViT UCF101 fine-tuning (single GPU)")

    # Paths
    p.add_argument("--shvit-dir",   type=Path, default=Path("SHViT"))
    p.add_argument("--finetune",    type=Path, default=Path("weights/shvit_s4.pth"),
                   help="ImageNet pretrained checkpoint to start from")
    p.add_argument("--data-root",   type=Path, default=Path("data"))
    p.add_argument("--output-dir",  type=Path,
                   default=Path("CV_Research_Paper_UCF101/Stage 3: fine-tuning SHViT/shvit_s4"))
    p.add_argument("--resume",      type=Path, default=None,
                   help="Resume a previous fine-tuning run from a checkpoint_N.pth")

    # Model — names are lowercase as registered by model/build.py @register_model
    p.add_argument("--model", default="shvit_s4",
                   choices=["shvit_s1", "shvit_s2", "shvit_s3", "shvit_s4"])
    p.add_argument("--input-size", default=224, type=int)

    # Training
    p.add_argument("--epochs",        default=30,   type=int)
    p.add_argument("--batch-size",    default=64,   type=int)
    p.add_argument("--lr",            default=1e-4, type=float,
                   help="Peak LR (no world-size scaling; paper used 1e-3 / 8 GPUs / bs-256)")
    p.add_argument("--min-lr",        default=1e-6, type=float)
    p.add_argument("--warmup-epochs", default=5,    type=int)
    p.add_argument("--weight-decay",  default=0.025, type=float)
    p.add_argument("--clip-grad",     default=0.02,  type=float,
                   help="Gradient norm clip threshold (0 = disabled)")
    p.add_argument("--seed",          default=0, type=int)
    p.add_argument("--num-workers",   default=2, type=int)
    p.add_argument("--save-freq",     default=10, type=int,
                   help="Save a resumable checkpoint every N epochs")

    # Eval-only mode
    p.add_argument("--eval", action="store_true",
                   help="Run evaluation only (requires --resume or --finetune with matching classes)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(args):
    splits.ensure_prepared(args.data_root)
    train_ds, val_ds = splits.load_split(
        args.data_root,
        train_transform=build_train_transform(img_size=args.input_size),
        val_transform=build_val_transform(img_size=args.input_size),
        input_size=args.input_size,
    )
    print(f"UCF101 split: {len(train_ds):,} train  {len(val_ds):,} val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=int(1.5 * args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model + checkpoint helpers
# ---------------------------------------------------------------------------

def load_pretrained(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load ImageNet weights, dropping head keys whose shape mismatches.
    Mirrors the --finetune branch in SHViT's main.py exactly."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    head_keys = [
        "head.l.weight", "head.l.bias",
        "head_dist.l.weight", "head_dist.l.bias",
    ]
    model_sd = model.state_dict()
    removed = []
    for k in head_keys:
        if k in state_dict and state_dict[k].shape != model_sd.get(k, torch.empty(0)).shape:
            del state_dict[k]
            removed.append(k)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights loaded from {ckpt_path}")
    print(f"  Removed (shape mismatch) : {removed}")
    if msg.missing_keys:
        print(f"  Missing keys           : {msg.missing_keys}")


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, args) -> float:
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, args, mixup_fn):
    model.train()
    total_loss, n = 0.0, 0

    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * samples.size(0)
        n += samples.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_outputs, all_targets_list = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        all_outputs.append(outputs.cpu())
        all_targets_list.append(targets.cpu())
        n += bs

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)
    top1 = top_k_accuracy(all_outputs, all_targets, k=1) / 100.0
    top5 = top_k_accuracy(all_outputs, all_targets, k=5) / 100.0
    return total_loss / n, top1, top5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    sys.path.insert(0, str(args.shvit_dir.resolve()))
    try:
        import model as _shvit_pkg  # noqa: F401 — registers shvit_s1..s4 with timm
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import SHViT `model` package from {args.shvit_dir}.\n"
            f"Make sure --shvit-dir points to the cloned SHViT repo.\n{exc}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    print(f"Model: {args.model}   Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training_log.csv"
    best_path = args.output_dir / "best.pth"

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader = build_loaders(args)

    # ---- Model --------------------------------------------------------------
    model = create_model(args.model, pretrained=False, num_classes=NUM_CLASSES)
    if args.finetune and args.finetune.exists():
        load_pretrained(model, args.finetune)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Loss / Mixup -------------------------------------------------------
    mixup_fn = build_mixup_fn(num_classes=NUM_CLASSES)
    criterion = SoftTargetCrossEntropy()

    # ---- Optimizer / scaler -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ---- Resume -------------------------------------------------------------
    start_epoch = 0
    best_top1 = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  (best top-1: {best_top1*100:.2f}%)")

    # ---- Eval only ----------------------------------------------------------
    if args.eval:
        _, top1, top5 = evaluate_epoch(model, val_loader, device)
        print(f"Eval  top-1: {top1*100:.2f}%   top-5: {top5*100:.2f}%")
        return

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "val_loss", "val_top1", "val_top5", "time_sec"]
            )

    # ---- Training loop ------------------------------------------------------
    print(f"\nFine-tuning for {args.epochs} epochs (starting at {start_epoch})")
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(epoch, args)
        set_lr(optimizer, lr)

        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, args, mixup_fn,
        )
        val_loss, val_top1, val_top5 = evaluate_epoch(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        print(
            f"[{epoch+1:3d}/{args.epochs}] lr={lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"top1={val_top1*100:.2f}%  top5={val_top5*100:.2f}%  ({elapsed:.0f}s)"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{lr:.6e}",
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{val_top1:.4f}", f"{val_top5:.4f}",
                f"{elapsed:.1f}",
            ])

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_top1": val_top1, "val_top5": val_top5}, best_path)
            print(f"  -> new best ({val_top1*100:.2f}%), saved to {best_path}")

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            ckpt_path = args.output_dir / f"checkpoint_{epoch+1:03d}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_top1": best_top1,
                "args": vars(args),
            }, ckpt_path)

    print(f"\nDone.  Best val top-1: {best_top1*100:.2f}%")
    print(f"Log  : {csv_path}")
    print(f"Best : {best_path}")


if __name__ == "__main__":
    main()
