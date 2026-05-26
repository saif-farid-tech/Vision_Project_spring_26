"""
eda.py
Flowers102 EDA visualizations for the report's data section.

Outputs (under --output-dir):
    eda_samples.png            5x5 grid of random training images + labels
    eda_image_sizes.png        width / height histograms (n=1000 random samples)
    eda_class_distribution.png stacked class-frequency bars (alphabetical)
    eda_summary.txt            numerical statistics

Usage:
    python eda.py --data-root data \\
        --output-dir CV_Research_Paper_Flowers102/Stage\\ 4:\\ Benchmarking\\ and\\ Demo/analysis/eda_outputs
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import splits  # noqa: E402
from datasets.flowers102 import OxfordFlowers  # noqa: E402


def load_datasets(data_root: Path):
    """Returns (train_items, test_items, classnames) from the Tip-Adapter split."""
    splits.ensure_prepared(data_root)
    ds = OxfordFlowers(root=str(data_root), num_shots=-1)
    return ds.train_full, ds.test, ds.classnames


def class_counts(items) -> Counter:
    return Counter(item.label for item in items)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sample_grid(items, classes, out_path: Path, n_rows=5, n_cols=5):
    indices = random.sample(range(len(items)), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
    for ax, idx in zip(axes.flat, indices):
        item = items[idx]
        img = Image.open(item.impath).convert("RGB")
        ax.imshow(img)
        ax.set_title(classes[item.label].replace("_", " "), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Flowers102 random training samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_image_sizes(sizes, out_path: Path):
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=40, color="steelblue", edgecolor="black")
    axes[0].set_title(f"Image widths  (n={len(widths)})")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[1].hist(heights, bins=40, color="darkorange", edgecolor="black")
    axes[1].set_title(f"Image heights (n={len(heights)})")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(train, test, classes, out_path: Path):
    train_counts = class_counts(train)
    test_counts = class_counts(test)
    order = list(range(len(classes)))
    train_vals = [train_counts.get(i, 0) for i in order]
    test_vals = [test_counts.get(i, 0) for i in order]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, train_vals, color="steelblue", label="train")
    ax.bar(x, test_vals, color="darkorange", bottom=train_vals, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels([classes[i].replace("_", " ") for i in order],
                       rotation=90, fontsize=6)
    ax.set_ylabel("Image count")
    ax.set_title("Flowers102 class distribution (alphabetical)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(train, test, sizes, classes, out_path: Path):
    train_per = list(class_counts(train).values())
    test_per = list(class_counts(test).values())
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]

    lines = [
        "Flowers102 EDA Summary",
        "=" * 40,
        f"Total images        : {len(train) + len(test):,}",
        f"  train             : {len(train):,}",
        f"  test              : {len(test):,}",
        f"Number of classes   : {len(classes)}",
        "",
        "Train per-class counts",
        f"  min / max / mean  : {min(train_per)} / {max(train_per)} / {np.mean(train_per):.2f}",
        f"  std               : {np.std(train_per):.4f}",
        "Test per-class counts",
        f"  min / max / mean  : {min(test_per)} / {max(test_per)} / {np.mean(test_per):.2f}",
        f"  std               : {np.std(test_per):.4f}",
        "",
        f"Image dimensions (random sample of {len(sizes)} train images)",
        f"  width  min/max    : {min(widths)} / {max(widths)}",
        f"  width  mean/median: {np.mean(widths):.1f} / {np.median(widths):.1f}",
        f"  height min/max    : {min(heights)} / {max(heights)}",
        f"  height mean/median: {np.mean(heights):.1f} / {np.median(heights):.1f}",
        "",
        "Class list (sorted):",
        ", ".join(classes),
    ]
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CV_Research_Paper_Flowers102/Stage 4: Benchmarking and Demo/analysis/eda_outputs"),
    )
    p.add_argument("--size-sample", type=int, default=1000,
                   help="number of train images sampled for the size histograms")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, classes = load_datasets(args.data_root)
    print(f"Train: {len(train):,}  Test: {len(test):,}  Classes: {len(classes)}")

    # 5x5 sample grid
    plot_sample_grid(train, classes, args.output_dir / "eda_samples.png")
    print("Wrote eda_samples.png")

    # Image-size histograms — header-only reads
    n = min(args.size_sample, len(train))
    indices = random.sample(range(len(train)), n)
    sizes = []
    for i in indices:
        with Image.open(train[i].impath) as im:
            sizes.append(im.size)  # (width, height)
    plot_image_sizes(sizes, args.output_dir / "eda_image_sizes.png")
    print("Wrote eda_image_sizes.png")

    # Class-frequency bars (alphabetical)
    plot_class_distribution(train, test, classes,
                            args.output_dir / "eda_class_distribution.png")
    print("Wrote eda_class_distribution.png")

    # Summary
    write_summary(train, test, sizes, classes,
                  args.output_dir / "eda_summary.txt")
    print("Wrote eda_summary.txt")


if __name__ == "__main__":
    main()
