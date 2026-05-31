"""
error_analysis.py
Detailed error analysis for SHViT-S4 on the Food-101 test set.

Reads <results-dir>/shvit_s4/results.json (per_class_acc, confusion_matrix,
all_preds, all_targets) and produces:

    confusion_top20.png         row-normalized heatmap, 20 worst classes (seaborn)
    worst_15_categories.png     horizontal bar chart of the 15 lowest accuracies
    misclassified_grid.png      4x4 grid of random misclassified test images
                                annotated with true / predicted / confidence
    error_analysis_summary.txt  worst-10, best-10, and top-10 confused pairs

Usage:
    python error_analysis.py \\
        --results-dir eval_outputs \\
        --data-root   data \\
        --shvit-dir   SHViT \\
        --checkpoint  "Stage 3: fine-tuning SHViT/shvit_s4/best.pth" \\
        --output-dir  error_analysis
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augmentation import build_val_transform                                  # noqa: E402
from tip_datasets import read_image                                           # noqa: E402
import splits                                                                 # noqa: E402


NUM_CLASSES        = 101
N_WORST_HEATMAP    = 20
N_WORST_BARS       = 15
N_GRID_ROWS        = 4
N_GRID_COLS        = 4
N_TOP_CONFUSED     = 10


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_shvit_s4(shvit_dir: Path, checkpoint: Path, device):
    """Build SHViT-S4 via timm and load the fine-tune checkpoint."""
    sys.path.insert(0, str(shvit_dir.resolve()))
    import model as _shvit_pkg  # noqa: F401  registers shvit_* with timm
    from timm.models import create_model

    m = create_model("shvit_s4", pretrained=False, num_classes=NUM_CLASSES)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    msg = m.load_state_dict(sd, strict=False)
    if msg.missing_keys:
        print(f"  [model] missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"  [model] unexpected keys: {len(msg.unexpected_keys)}")
    m.eval().to(device)
    return m


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def confused_pairs(cm: np.ndarray, class_names, top_k: int = N_TOP_CONFUSED):
    """Top-k off-diagonal (i, j) pairs by P(predicted = j | true = i)."""
    row_sums = cm.sum(axis=1)
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        if row_sums[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            count = cm[i, j]
            if count == 0:
                continue
            rate = count / row_sums[i]
            pairs.append((rate, int(count), i, j))
    pairs.sort(reverse=True)
    return [(rate, count, class_names[i], class_names[j])
            for rate, count, i, j in pairs[:top_k]]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion_top20(cm: np.ndarray, per_class_acc, class_names, out_path: Path):
    n = N_WORST_HEATMAP
    worst_idx = np.argsort(per_class_acc)[:n]
    sub_labels = [class_names[i].replace("_", " ") for i in worst_idx]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.where(row_sums == 0, 1, row_sums)
    sub = cm_norm[np.ix_(worst_idx, worst_idx)]

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        sub, annot=True, fmt=".2f", cmap="Reds", vmin=0.0,
        xticklabels=sub_labels, yticklabels=sub_labels,
        cbar_kws={"label": "P(predicted | true)"},
        square=True, annot_kws={"size": 7}, ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion among {n} worst-performing classes "
                 "(rows = true; values are P(predicted | true))")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_worst_15(per_class_acc, class_names, out_path: Path):
    idx = np.argsort(per_class_acc)[:N_WORST_BARS]
    labels = [class_names[i].replace("_", " ") for i in idx]
    values = [per_class_acc[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(labels))
    ax.barh(y, values, color="indianred", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class accuracy (%)")
    ax.set_title(f"{N_WORST_BARS} worst-performing classes (SHViT-S4)")
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) + 8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_misclassified_grid(args, all_preds, all_targets, class_names, test_datums,
                            device, out_path: Path):
    misclass = np.where(all_preds != all_targets)[0]
    if misclass.size == 0:
        print("[skip] misclassified_grid: zero misclassifications")
        return
    n = N_GRID_ROWS * N_GRID_COLS
    sample = sorted(random.sample(list(misclass), min(n, len(misclass))))

    # test_datums is the ordered Zhou-split test set; its order matches the
    # all_preds / all_targets order produced by evaluate_all.py (shuffle=False).
    val_tf = build_val_transform()

    model = load_shvit_s4(args.shvit_dir, args.checkpoint, device)

    inputs = torch.stack(
        [val_tf(read_image(test_datums[int(i)].impath)) for i in sample]
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(model(inputs), dim=1).cpu()

    fig, axes = plt.subplots(N_GRID_ROWS, N_GRID_COLS,
                             figsize=(N_GRID_COLS * 3.6, N_GRID_ROWS * 4.0))
    for ax, batch_pos, idx in zip(axes.flat, range(len(sample)), sample):
        img = read_image(test_datums[int(idx)].impath)
        true_c = int(all_targets[idx])
        pred_c = int(all_preds[idx])         # canonical pred from results.json
        conf   = probs[batch_pos, pred_c].item()
        ax.imshow(img)
        ax.set_title(
            f"true: {class_names[true_c].replace('_', ' ')}\n"
            f"pred: {class_names[pred_c].replace('_', ' ')}  "
            f"({conf*100:.1f}%)",
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(sample):]:
        ax.set_visible(False)
    fig.suptitle("Random misclassified test images (SHViT-S4)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(results, class_names, top_pairs, test_size, out_path: Path):
    pca = results["per_class_acc"]
    order = np.argsort(pca)
    worst10 = order[:10]
    best10  = order[::-1][:10]

    lines = [
        "SHViT-S4 Error Analysis Summary",
        "=" * 48,
        "",
        "Overall:",
        f"  - Test set size       : {test_size:,} images",
        f"  - Test top-1 accuracy : {results['top1_acc']:.2f}%",
        f"  - Test top-5 accuracy : {results['top5_acc']:.2f}%",
        "",
        "Worst-10 classes by per-class accuracy:",
    ]
    for rank, idx in enumerate(worst10, 1):
        lines.append(f"  {rank:>2}. {class_names[idx]:<28} {pca[idx]:6.2f}%")
    lines += ["", "Best-10 classes (for context):"]
    for rank, idx in enumerate(best10, 1):
        lines.append(f"  {rank:>2}. {class_names[idx]:<28} {pca[idx]:6.2f}%")
    lines += ["", "Top-10 most confused class pairs (true -> predicted):"]
    for rank, (rate, count, t, p) in enumerate(top_pairs, 1):
        lines.append(
            f"  {rank:>2}. {t:<28} -> {p:<28}  "
            f"{rate*100:5.2f}%  (n={count})"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("eval_outputs"))
    p.add_argument("--data-root",   type=Path, default=Path("data"))
    p.add_argument("--shvit-dir",   type=Path, default=_REPO_ROOT / "SHViT")
    p.add_argument("--checkpoint",  type=Path,
                   default=_REPO_ROOT / "Stage 3: fine-tuning SHViT/shvit_s4/best.pth")
    p.add_argument("--output-dir",  type=Path, default=Path("error_analysis"))
    p.add_argument("--seed",        type=int,  default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load results.json ----------------------------------------------
    results_path = args.results_dir / "shvit_s4" / "results.json"
    if not results_path.exists():
        raise SystemExit(f"results.json not found at {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    pca         = np.asarray(results["per_class_acc"], dtype=float)
    cm          = np.asarray(results["confusion_matrix"])
    all_preds   = np.asarray(results["all_preds"])
    all_targets = np.asarray(results["all_targets"])

    # ---- Class names + test set from the Tip-Adapter Zhou split ----------
    base = splits.build_food101(args.data_root)
    class_names = list(base.classnames)
    test_datums = base.test
    test_size = len(test_datums)
    if len(class_names) != NUM_CLASSES:
        raise SystemExit(f"expected {NUM_CLASSES} classes, got {len(class_names)}")

    # ---- Console: worst-10 / best-10 ------------------------------------
    order = np.argsort(pca)
    print("\nWorst-10 classes:")
    for idx in order[:10]:
        print(f"  {class_names[idx]:<28} {pca[idx]:6.2f}%")
    print("\nBest-10 classes:")
    for idx in order[::-1][:10]:
        print(f"  {class_names[idx]:<28} {pca[idx]:6.2f}%")

    # ---- Console: most confused pairs -----------------------------------
    top_pairs = confused_pairs(cm, class_names, top_k=N_TOP_CONFUSED)
    print("\nTop-10 most confused class pairs (true -> predicted):")
    for rate, count, t, p in top_pairs:
        print(f"  {t:<28} -> {p:<28}  {rate*100:5.2f}%  (n={count})")

    # ---- Figures ---------------------------------------------------------
    plot_confusion_top20(cm, pca, class_names,
                         args.output_dir / "confusion_top20.png")
    plot_worst_15(pca, class_names,
                  args.output_dir / "worst_15_categories.png")

    if not args.checkpoint.exists():
        print(f"[warn] checkpoint not at {args.checkpoint}; "
              "skipping misclassified_grid.png")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            plot_misclassified_grid(
                args, all_preds, all_targets, class_names, test_datums, device,
                args.output_dir / "misclassified_grid.png",
            )
        except Exception as exc:
            print(f"[warn] misclassified_grid failed: {exc}")

    # ---- Summary ---------------------------------------------------------
    write_summary(results, class_names, top_pairs, test_size,
                  args.output_dir / "error_analysis_summary.txt")


if __name__ == "__main__":
    main()
