"""
demo.py
Single-image inference with SHViT-S4 fine-tuned on Food-101.

Loads the checkpoint, preprocesses the image with the same val transform
used in training (Resize 256 -> CenterCrop 224 -> ImageNet normalize),
prints top-5 predictions, and saves the image with the top prediction
overlaid.

Usage:
    python demo.py --image food.jpg \\
        --checkpoint "Stage 3: fine-tuning SHViT/shvit_s4/best.pth" \\
        --shvit-dir  SHViT \\
        --data-root  data
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import Food101


_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augmentation import build_val_transform  # noqa: E402


NUM_CLASSES = 101


def load_shvit_s4(shvit_dir: Path, checkpoint: Path, device):
    sys.path.insert(0, str(shvit_dir.resolve()))
    import model as _shvit_pkg  # noqa: F401  registers shvit_* with timm
    from timm.models import create_model

    m = create_model("shvit_s4", pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    msg = m.load_state_dict(sd, strict=False)
    if msg.missing_keys:
        print(f"  [model] missing keys   : {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"  [model] unexpected keys: {len(msg.unexpected_keys)}")
    m.eval().to(device)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--shvit-dir",  type=Path, required=True)
    p.add_argument("--data-root",  type=Path, required=True,
                   help="Food-101 root (used only to fetch the 101 class names)")
    p.add_argument("--output",     type=Path, default=Path("demo_output.png"))
    p.add_argument("--top-k",      type=int,  default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Class names ----------------------------------------------------
    class_names = list(
        Food101(root=str(args.data_root), split="test", download=True).classes
    )

    # ---- Image + preprocessing -----------------------------------------
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    img_pil = Image.open(args.image).convert("RGB")
    transform = build_val_transform(img_size=224)
    x = transform(img_pil).unsqueeze(0).to(device)

    # ---- Model ---------------------------------------------------------
    model = load_shvit_s4(args.shvit_dir, args.checkpoint, device)

    # ---- Inference -----------------------------------------------------
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0].cpu()

    top_probs, top_idx = probs.topk(args.top_k)
    top_probs = top_probs.tolist()
    top_idx   = top_idx.tolist()

    print(f"\nTop-{args.top_k} predictions for {args.image.name}:")
    for rank, (prob, idx) in enumerate(zip(top_probs, top_idx), 1):
        print(f"  {rank}. {class_names[idx]:<28} {prob * 100:6.2f}%")

    # ---- Overlay + save ------------------------------------------------
    top_label = class_names[top_idx[0]].replace("_", " ")
    top_conf  = top_probs[0] * 100.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_pil)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02, 0.98,
        f"{top_label}\n{top_conf:.1f}%",
        transform=ax.transAxes,
        fontsize=14, va="top", ha="left", color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=8, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
