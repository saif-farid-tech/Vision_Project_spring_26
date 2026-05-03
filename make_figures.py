"""
make_figures.py
Generate the report's headline figures from training logs and evaluate_all.py
results. Produces (under --output-dir):

    fig_training_curves.png   val top-1 (%) over epochs, all 6 models
    fig_train_loss.png        train loss over epochs, all 6 models
    fig_speed_vs_accuracy.png GPU throughput vs. test top-1 (headline figure)
    fig_accuracy_bars.png     test top-1 bars, sorted descending

Usage:
    python make_figures.py --results-dir eval_outputs --logs-dir . --output-dir figures
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_REPO_ROOT = Path(__file__).parent

# (name, kind, subdir relative to logs-dir, display label, plot color)
# Default tab10 colors give 6 distinct hues out of the box.
MODELS = [
    ("resnet50",     "baseline", "Stage 2: baseline models/resnet50",     "ResNet-50",   "C0"),
    ("mobilenet_v2", "baseline", "Stage 2: baseline models/mobilenet_v2", "MobileNetV2", "C1"),
    ("shvit_s1",     "shvit",    "Stage 3: fine-tuning SHViT/shvit_s1",   "SHViT-S1",    "C2"),
    ("shvit_s2",     "shvit",    "Stage 3: fine-tuning SHViT/shvit_s2",   "SHViT-S2",    "C3"),
    ("shvit_s3",     "shvit",    "Stage 3: fine-tuning SHViT/shvit_s3",   "SHViT-S3",    "C4"),
    ("shvit_s4",     "shvit",    "Stage 3: fine-tuning SHViT/shvit_s4",   "SHViT-S4",    "C5"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_log(path: Path):
    """CSV columns: epoch,lr,train_loss,val_loss,val_top1,val_top5,time_sec.
    val_top1/val_top5 are stored as fractions in [0, 1] by both training scripts.
    """
    cols = {"epoch": [], "train_loss": [], "val_top1": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            cols["epoch"].append(int(row["epoch"]))
            cols["train_loss"].append(float(row["train_loss"]))
            cols["val_top1"].append(float(row["val_top1"]))
    return cols


def load_results(path: Path):
    with open(path) as f:
        return json.load(f)


def gather(args):
    rows = []
    for name, kind, subdir, label, color in MODELS:
        log_path = args.logs_dir    / subdir / "training_log.csv"
        res_path = args.results_dir / name   / "results.json"

        log = load_log(log_path)         if log_path.exists() else None
        res = load_results(res_path)     if res_path.exists() else None
        if log is None:
            print(f"[warn] {name}: no training log at {log_path}")
        if res is None:
            print(f"[warn] {name}: no results.json at {res_path}")

        rows.append({
            "name": name, "kind": kind, "label": label, "color": color,
            "log": log, "results": res,
        })
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _line_style(kind: str) -> str:
    return "-" if kind == "baseline" else "--"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_training_curves(rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for m in rows:
        if m["log"] is None:
            continue
        y = [v * 100.0 for v in m["log"]["val_top1"]]   # fraction -> %
        ax.plot(m["log"]["epoch"], y, _line_style(m["kind"]),
                color=m["color"], label=m["label"], linewidth=2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print(f"[skip] {out_path}: no logs available")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 accuracy (%)")
    ax.set_title("Validation top-1 over training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, title="Model")
    _save(fig, out_path)


def fig_train_loss(rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for m in rows:
        if m["log"] is None:
            continue
        ax.plot(m["log"]["epoch"], m["log"]["train_loss"], _line_style(m["kind"]),
                color=m["color"], label=m["label"], linewidth=2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print(f"[skip] {out_path}: no logs available")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, title="Model")
    _save(fig, out_path)


def fig_speed_vs_accuracy(rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    plotted = 0
    for m in rows:
        if m["results"] is None:
            continue
        x = m["results"].get("gpu_throughput_imgs_s")
        y = m["results"].get("top1_acc")
        if x is None or y is None:
            print(f"[warn] {m['name']}: missing throughput or top-1 in results.json")
            continue
        marker = "o" if m["kind"] == "shvit" else "s"
        ax.scatter(x, y, marker=marker, s=160, color=m["color"],
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(m["label"], (x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=10, zorder=4)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print(f"[skip] {out_path}: no results available")
        return

    shape_legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="Baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="SHViT"),
    ]
    ax.legend(handles=shape_legend, loc="lower right", frameon=True)
    ax.set_xlabel("GPU throughput (images/s, batch 64)")
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_title("Speed vs. accuracy on Food-101 test set")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def fig_accuracy_bars(rows, out_path: Path):
    have = [m for m in rows
            if m["results"] is not None
            and m["results"].get("top1_acc") is not None]
    if not have:
        print(f"[skip] {out_path}: no results available")
        return
    have.sort(key=lambda m: m["results"]["top1_acc"], reverse=True)

    labels = [m["label"]                    for m in have]
    values = [m["results"]["top1_acc"]      for m in have]
    colors = [m["color"]                    for m in have]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    headroom = max(values) * 0.10
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + headroom * 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_title("Test top-1 accuracy on Food-101 (sorted)")
    ax.set_ylim(0, max(values) + headroom)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("eval_outputs"),
                   help="Directory with per-model results.json from evaluate_all.py")
    p.add_argument("--logs-dir",    type=Path, default=_REPO_ROOT,
                   help="Repo root containing the Stage 2 / Stage 3 log subdirectories")
    p.add_argument("--output-dir",  type=Path, default=Path("figures"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = gather(args)

    fig_training_curves  (rows, args.output_dir / "fig_training_curves.png")
    fig_train_loss       (rows, args.output_dir / "fig_train_loss.png")
    fig_speed_vs_accuracy(rows, args.output_dir / "fig_speed_vs_accuracy.png")
    fig_accuracy_bars    (rows, args.output_dir / "fig_accuracy_bars.png")


if __name__ == "__main__":
    main()
