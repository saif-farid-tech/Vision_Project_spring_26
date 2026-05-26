"""
verify_model.py
Loads SHViT-S4 from a local checkpoint and runs inference on a small
subset of Caltech-101 validation images to confirm the model loads correctly.

Usage:
    python verify_model.py \\
        --shvit-dir SHViT \\
        --checkpoint weights/shvit_s4.pth \\
        --data-root data \\
        --num-images 50

The script prints per-image predictions and a summary timing.
Top-1 predictions are ImageNet class indices, so expect ~zero accuracy vs.
Caltech-101 labels — the point is just to confirm the model runs without errors.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in [str(_SCRIPT_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transform(img_size: int = 224) -> transforms.Compose:
    """CLIP / Tip-Adapter normalization, bicubic resize."""
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def collect_images(image_dir: Path, n: int):
    """Return up to n (path, class_name) pairs from class subdirectories."""
    items = []
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name in ("BACKGROUND_Google", "Faces_easy"):
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            items.append((img_path, class_dir.name))
            if len(items) >= n:
                return items
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SHViT-S4 loads and runs")
    parser.add_argument("--shvit-dir", type=Path, default=Path("SHViT"),
                        help="Path to cloned SHViT repo")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/shvit_s4.pth"),
                        help="Path to shvit_s4.pth")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Caltech-101 parent dir; expects "
                             "<data-root>/caltech-101/101_ObjectCategories/")
    parser.add_argument("--num-images", type=int, default=50,
                        help="Number of images to run inference on")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Add SHViT repo to path so we can import its models package
    # ------------------------------------------------------------------ #
    sys.path.insert(0, str(args.shvit_dir.resolve()))
    try:
        import model as _shvit_pkg  # noqa: F401 — registers shvit_s* with timm
        import timm
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import SHViT `model` package: {exc}\n"
                 f"Make sure --shvit-dir points to the cloned repo "
                 f"and requirements are installed.")

    # ------------------------------------------------------------------ #
    # 2. Build model
    # ------------------------------------------------------------------ #
    print("Building shvit_s4 ...")
    net = timm.create_model("shvit_s4", pretrained=False, num_classes=1000)

    # ------------------------------------------------------------------ #
    # 3. Load checkpoint
    # ------------------------------------------------------------------ #
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}\n"
                 f"Download it with:\n"
                 f"  wget https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth "
                 f"-O {args.checkpoint}")

    print(f"Loading weights from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("  Weights loaded successfully.")

    net.to(args.device)
    net.eval()

    # ------------------------------------------------------------------ #
    # 4. Collect images (auto-prepare dataset if missing)
    # ------------------------------------------------------------------ #
    image_dir = args.data_root / "caltech-101" / "101_ObjectCategories"
    if not image_dir.exists():
        print(f"[info] {image_dir} not found, downloading Caltech-101 first ...")
        import splits  # noqa: E402
        splits.ensure_prepared(args.data_root)

    items = collect_images(image_dir, args.num_images)
    if not items:
        sys.exit(f"[ERROR] No .jpg images found under {image_dir}")

    print(f"\nRunning inference on {len(items)} images (device={args.device}) ...")
    transform = get_transform()

    t0 = time.perf_counter()
    results = []
    with torch.no_grad():
        for img_path, true_class in items:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(args.device)
            logits = net(x)
            pred_idx = int(logits.argmax(dim=1).item())
            results.append((img_path.name, true_class, pred_idx))

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 5. Print summary
    # ------------------------------------------------------------------ #
    print(f"\n{'Image':<30} {'True class':<25} {'Pred ImageNet idx':>16}")
    print("-" * 75)
    for name, cls, pred in results[:20]:
        print(f"{name:<30} {cls:<25} {pred:>16}")
    if len(results) > 20:
        print(f"  ... ({len(results) - 20} more)")

    print(f"\nTotal inference time : {elapsed:.2f}s  "
          f"({elapsed / len(results) * 1000:.1f} ms/image)")
    print("\n[OK] Model loaded and ran inference without errors.")
    print("Note: top-1 predictions are ImageNet class indices — accuracy vs.")
    print("Caltech-101 labels is expected to be low without fine-tuning.")


if __name__ == "__main__":
    main()
