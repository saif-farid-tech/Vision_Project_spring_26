"""
evaluate_all.py
Run inference and benchmarks for the 6 trained Food-101 classifiers
(ResNet-50, MobileNetV2, SHViT-S1..S4) on Food101(split='test').

Per model:
    - Top-1 / Top-5 accuracy
    - Per-class accuracy (101 floats) and 101x101 confusion matrix
    - all_preds and all_targets
    - Param count (M), GFLOPs (fvcore), GPU throughput, CPU latency

Outputs:
    <output-dir>/<model>/results.json
    <output-dir>/results_table.md
    <output-dir>/results_table.tex   (sorted by GFLOPs ascending)

Usage:
    python evaluate_all.py \\
        --checkpoints-dir . \\
        --shvit-dir       SHViT \\
        --data-root       data \\
        --output-dir      eval_outputs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.models import mobilenet_v2, resnet50

_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augmentation import build_val_transform                                # noqa: E402
from metrics import evaluate_model                                          # noqa: E402


NUM_CLASSES = 101

# (model name, subdir under checkpoints-dir, checkpoint format)
MODELS = [
    ("resnet50",     "Stage 2: baseline models/resnet50",     "baseline"),
    ("mobilenet_v2", "Stage 2: baseline models/mobilenet_v2", "baseline"),
    ("shvit_s1",     "Stage 3: fine-tuning SHViT/shvit_s1",   "shvit"),
    ("shvit_s2",     "Stage 3: fine-tuning SHViT/shvit_s2",   "shvit"),
    ("shvit_s3",     "Stage 3: fine-tuning SHViT/shvit_s3",   "shvit"),
    ("shvit_s4",     "Stage 3: fine-tuning SHViT/shvit_s4",   "shvit"),
]


# ---------------------------------------------------------------------------
# Model + checkpoint loading
# ---------------------------------------------------------------------------

def build_model(name: str) -> nn.Module:
    if name == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(2048, NUM_CLASSES)
    elif name == "mobilenet_v2":
        m = mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    elif name.startswith("shvit_"):
        from timm.models import create_model
        m = create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def load_state_dict(path: Path, kind: str):
    """
    baseline: bare state_dict on disk (torch.save(model.state_dict(), path))
    shvit   : {"model": state_dict, "optimizer": ..., "epoch": ...}
    Falls back gracefully if the file actually wraps the state_dict.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "shvit":
        return obj["model"]
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model" in obj:
            return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return obj


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def measure_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_gflops(name: str, img_size: int = 224):
    """fvcore FlopCountAnalysis on a fresh CPU instance (1-image input)."""
    from fvcore.nn import FlopCountAnalysis
    m = build_model(name).cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    fca = FlopCountAnalysis(m, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total() / 1e9


def measure_gpu_throughput(model, device, batch=64, warmup=20, iters=200, img_size=224):
    model = model.to(device).eval()
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return iters * batch / elapsed  # images/sec


def measure_cpu_latency(model, batch=1, warmup=10, iters=100, img_size=224):
    model = model.to("cpu").eval()
    x = torch.randn(batch, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        elapsed = time.perf_counter() - t0
    return (elapsed / iters) * 1000.0  # ms per forward pass


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(v, spec):
    return f"{v:{spec}}" if v is not None else "-"


def write_markdown_table(rows, path: Path) -> None:
    headers = ["Model", "Params (M)", "GFLOPs", "Top-1 (%)", "Top-5 (%)",
               "GPU (img/s)", "CPU (ms)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"],
            _fmt(r["params_M"],       ".2f"),
            _fmt(r["gflops"],         ".2f"),
            _fmt(r["top1"],           ".2f"),
            _fmt(r["top5"],           ".2f"),
            _fmt(r["gpu_throughput"], ".0f"),
            _fmt(r["cpu_latency_ms"], ".2f"),
        ]) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex_table(rows, path: Path) -> None:
    header = ("Model & Params (M) & GFLOPs & Top-1 (\\%) & Top-5 (\\%) & "
              "GPU (img/s) & CPU (ms) \\\\")
    body = []
    for r in rows:
        body.append(" & ".join([
            r["model"].replace("_", r"\_"),
            _fmt(r["params_M"],       ".2f"),
            _fmt(r["gflops"],         ".2f"),
            _fmt(r["top1"],           ".2f"),
            _fmt(r["top5"],           ".2f"),
            _fmt(r["gpu_throughput"], ".0f"),
            _fmt(r["cpu_latency_ms"], ".2f"),
        ]) + r" \\")
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        header,
        r"\hline",
        *body,
        r"\hline",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path, default=_REPO_ROOT)
    p.add_argument("--shvit-dir",       type=Path, default=_REPO_ROOT / "SHViT")
    p.add_argument("--data-root",       type=Path, default=Path("data"))
    p.add_argument("--output-dir",      type=Path, default=Path("eval_outputs"))
    p.add_argument("--batch-size",      type=int,  default=64)
    p.add_argument("--num-workers",     type=int,  default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # Register SHViT model family with timm via side-effect import.
    sys.path.insert(0, str(args.shvit_dir.resolve()))
    try:
        import model as _shvit_pkg  # noqa: F401  registers shvit_s1..s4
    except ImportError as exc:
        print(f"[warn] SHViT model package not importable from {args.shvit_dir}: {exc}\n"
              f"       Any SHViT checkpoints will be skipped.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Test set --------------------------------------------------------
    test_ds = Food101(
        root=str(args.data_root), split="test",
        transform=build_val_transform(), download=True,
    )
    class_names = test_ds.classes
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test set: {len(test_ds):,} images")

    # ---- Per-model loop --------------------------------------------------
    summary_rows = []
    for name, subdir, kind in MODELS:
        ckpt = args.checkpoints_dir / subdir / "best.pth"
        if not ckpt.exists():
            print(f"\n[skip] {name}: no checkpoint at {ckpt}")
            continue

        print(f"\n=== {name} ===")
        try:
            model = build_model(name)
        except Exception as exc:
            print(f"[skip] {name}: model build failed ({exc})")
            continue

        try:
            sd = load_state_dict(ckpt, kind)
        except Exception as exc:
            print(f"[skip] {name}: checkpoint load failed ({exc})")
            continue
        msg = model.load_state_dict(sd, strict=False)
        if msg.missing_keys:
            print(f"  missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"  unexpected keys: {len(msg.unexpected_keys)}")

        # ---- Inference ---------------------------------------------------
        eval_out = evaluate_model(model, test_loader, device, class_names)
        top1     = float(eval_out["top1_acc"])
        top5     = float(eval_out["top5_acc"])
        pca_dict = eval_out["per_class_acc"]              # {name: pct or None}
        pca_list = [pca_dict[c] for c in class_names]     # 101 floats (in class-index order)
        cm       = eval_out["confusion_matrix"]           # np.ndarray (101, 101)
        preds    = eval_out["all_preds"]
        targets  = eval_out["all_targets"]

        # ---- Benchmarks --------------------------------------------------
        params_m = measure_params_m(model)
        try:
            gflops = measure_gflops(name)
        except Exception as exc:
            print(f"  GFLOPs failed: {exc}")
            gflops = None

        gpu_thru = None
        if device.type == "cuda":
            try:
                gpu_thru = measure_gpu_throughput(model, device,
                                                  batch=64, warmup=20, iters=200)
                print(f"  GPU throughput @bs64: {gpu_thru:.0f} img/s")
            except Exception as exc:
                print(f"  GPU benchmark failed: {exc}")

        try:
            cpu_lat = measure_cpu_latency(model, batch=1, warmup=10, iters=100)
            print(f"  CPU latency @bs1   : {cpu_lat:.2f} ms")
        except Exception as exc:
            print(f"  CPU benchmark failed: {exc}")
            cpu_lat = None

        # ---- Per-model JSON ---------------------------------------------
        model_dir = args.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "model":                 name,
            "checkpoint":            str(ckpt),
            "top1_acc":              top1,
            "top5_acc":              top5,
            "per_class_acc":         pca_list,
            "confusion_matrix":      np.asarray(cm).tolist(),
            "all_preds":             np.asarray(preds).tolist(),
            "all_targets":           np.asarray(targets).tolist(),
            "params_M":              params_m,
            "gflops":                gflops,
            "gpu_throughput_imgs_s": gpu_thru,
            "cpu_latency_ms":        cpu_lat,
        }
        with open(model_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {model_dir/'results.json'}")

        summary_rows.append({
            "model":          name,
            "params_M":       params_m,
            "gflops":         gflops,
            "top1":           top1,
            "top5":           top5,
            "gpu_throughput": gpu_thru,
            "cpu_latency_ms": cpu_lat,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not summary_rows:
        print("\nNo models evaluated.")
        return

    # Sort by GFLOPs ascending; missing GFLOPs sink to the end.
    summary_rows.sort(key=lambda r: (r["gflops"] is None, r["gflops"] or 0.0))

    md_path  = args.output_dir / "results_table.md"
    tex_path = args.output_dir / "results_table.tex"
    write_markdown_table(summary_rows, md_path)
    write_latex_table(summary_rows, tex_path)
    print(f"\nWrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
