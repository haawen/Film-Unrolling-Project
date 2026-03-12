"""
Compare validation results across ALL model architectures:
  - nnU-Net 2D
  - nnU-Net 3D
  - 3D U-Net  (MONAI)
  - Swin UNETR (MONAI)

Reads summary.json from each experiment's validation folder and produces
tables + bar charts of Dice, IoU, Precision, Recall per class, as well as
training-curve overlays and a per-case box-plot comparison.

Usage:
    python Scripts/compare_results.py                   # auto-detect all experiments
    python Scripts/compare_results.py --save            # save figures to disk
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
NNUNET_RESULTS_BASE = BASE / "nnUNet_data" / "nnUNet_results"
MONAI_RESULTS_BASE = BASE / "monai_results"
OUTPUT_DIR = BASE / "visualizations"

# All dataset directories to scan
DATASET_NAMES = [
    "Dataset501_MickeyScroll",
    "Dataset502_MickeyScroll3D",
]

LABEL_NAMES = {0: "background", 1: "foreground_1", 2: "foreground_2"}
FG_LABELS = {k: v for k, v in LABEL_NAMES.items() if k != 0}

# Friendly short names for nnU-Net experiment directories
NNUNET_SHORT = {
    "nnUNetTrainerProgress__nnUNetPlans__2d": "nnU-Net 2D",
    "nnUNetTrainer__nnUNetPlans__2d": "nnU-Net 2D (default)",
    "nnUNetTrainerProgress__nnUNetPlans__3d_fullres": "nnU-Net 3D",
    "nnUNetTrainer__nnUNetPlans__3d_fullres": "nnU-Net 3D (default)",
}

# ─── Discovery ───────────────────────────────────────────────────────────────


def _scan_fold_dirs(parent: Path) -> dict[str, Path]:
    """Return {fold_name: summary_path} for fold_* dirs under *parent*."""
    folds = {}
    for fold_dir in sorted(parent.glob("fold_*")):
        summary = fold_dir / "validation" / "summary.json"
        if summary.exists():
            folds[fold_dir.name] = summary
    return folds


def discover_experiments() -> dict[str, dict]:
    """Find all experiments with at least one evaluated fold.
    Returns {display_name: {fold_name: summary_path}}."""
    experiments = {}

    for ds_name in DATASET_NAMES:
        # 1) nnU-Net experiments  (trainer__plans__config directories)
        nnunet_dir = NNUNET_RESULTS_BASE / ds_name
        if nnunet_dir.exists():
            for exp_dir in sorted(nnunet_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                folds = _scan_fold_dirs(exp_dir)
                if folds:
                    name = NNUNET_SHORT.get(exp_dir.name, exp_dir.name)
                    experiments[name] = folds

        # 2) MONAI experiments  (model-name directories)
        monai_dir = MONAI_RESULTS_BASE / ds_name
        if monai_dir.exists():
            for model_dir in sorted(monai_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                folds = _scan_fold_dirs(model_dir)
                if folds:
                    experiments[model_dir.name] = folds

    return experiments


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Metrics extraction ─────────────────────────────────────────────────────

CORE_METRICS = ["Dice", "IoU"]


def _safe(value, default=0.0):
    """Return *value* if it is a finite number, else *default*."""
    if value is None:
        return default
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def extract_mean_metrics(summary: dict) -> dict:
    """Return {class_label: {metric: value}} from the 'mean' key."""
    return summary.get("mean", {})


def extract_per_case_metrics(summary: dict) -> list[dict]:
    return summary.get("metric_per_case", [])


def aggregate_folds(fold_summaries: dict[str, Path]) -> dict:
    """Average metrics across folds → {class_label: {metric: mean_value}}."""
    all_means = []
    for path in fold_summaries.values():
        summary = load_summary(path)
        all_means.append(extract_mean_metrics(summary))

    if not all_means:
        return {}

    classes = list(all_means[0].keys())
    metrics = list(all_means[0][classes[0]].keys()) if classes else []

    aggregated = {}
    for cls in classes:
        aggregated[cls] = {}
        for metric in metrics:
            values = [_safe(m[cls].get(metric)) for m in all_means if cls in m]
            aggregated[cls][metric] = float(np.mean(values)) if values else 0.0
    return aggregated


def compute_precision_recall(agg: dict) -> dict:
    """Derive precision and recall from TP / FP / FN if available."""
    result = {}
    for cls, m in agg.items():
        tp = m.get("TP", 0)
        fp = m.get("FP", 0)
        fn = m.get("FN", 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        result[cls] = {"Precision": prec, "Recall": rec}
    return result


# ─── Display helpers ─────────────────────────────────────────────────────────


def _fg_mean(fold_paths: dict[str, Path]) -> dict[str, float]:
    """Return averaged foreground-mean Dice and IoU across folds."""
    fg = {}
    for path in fold_paths.values():
        summary = load_summary(path)
        fm = summary.get("foreground_mean", {})
        for metric in CORE_METRICS:
            fg.setdefault(metric, []).append(_safe(fm.get(metric)))
    return {m: float(np.mean(v)) for m, v in fg.items()}


def _model_params(fold_paths: dict[str, Path]) -> str:
    """Try to read n_parameters from any fold's summary."""
    for path in fold_paths.values():
        s = load_summary(path)
        n = s.get("n_parameters")
        if n is not None:
            return f"{n:,}"
    return "—"


# ─── Tables ──────────────────────────────────────────────────────────────────


def print_comparison_table(experiments: dict[str, dict]):
    """Print per-class Dice / IoU / Precision / Recall + foreground mean."""
    print("\n" + "=" * 100)
    print("  MODEL COMPARISON — Dataset501_MickeyScroll")
    print("=" * 100)

    header = (
        f"{'Model':<25} {'Class':<15} "
        f"{'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'Folds':>6}"
    )
    print(header)
    print("-" * len(header))

    for exp_name, fold_paths in experiments.items():
        agg = aggregate_folds(fold_paths)
        pr = compute_precision_recall(agg)
        n_folds = len(fold_paths)
        for cls in sorted(agg.keys()):
            cls_name = LABEL_NAMES.get(int(cls), f"class_{cls}")
            dice = _safe(agg[cls].get("Dice"))
            iou = _safe(agg[cls].get("IoU"))
            prec = _safe(pr.get(cls, {}).get("Precision"))
            rec = _safe(pr.get(cls, {}).get("Recall"))
            print(
                f"{exp_name:<25} {cls_name:<15} "
                f"{dice:>8.4f} {iou:>8.4f} {prec:>8.4f} {rec:>8.4f} {n_folds:>6}"
            )

    # Foreground means
    print("\n" + "-" * 100)
    print(f"{'Model':<25} {'Params':>15} {'FG Dice':>10} {'FG IoU':>10}")
    print("-" * 65)
    for exp_name, fold_paths in experiments.items():
        fm = _fg_mean(fold_paths)
        params = _model_params(fold_paths)
        print(
            f"{exp_name:<25} {params:>15} "
            f"{fm.get('Dice',0):>10.4f} {fm.get('IoU',0):>10.4f}"
        )
    print("=" * 100)


# ─── Plots ───────────────────────────────────────────────────────────────────

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


def plot_comparison(experiments: dict[str, dict], save: bool = False):
    """Grouped bar chart: Dice + IoU per class for each model."""
    exp_names = list(experiments.keys())
    classes = sorted(FG_LABELS.keys())
    n_exp = len(exp_names)
    n_cls = len(classes)

    dice_mat = np.zeros((n_exp, n_cls))
    iou_mat = np.zeros((n_exp, n_cls))
    for i, name in enumerate(exp_names):
        agg = aggregate_folds(experiments[name])
        for j, cls in enumerate(classes):
            dice_mat[i, j] = _safe(agg.get(str(cls), {}).get("Dice"))
            iou_mat[i, j] = _safe(agg.get(str(cls), {}).get("IoU"))

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 3 * n_exp), 5))
    x = np.arange(n_exp)
    width = 0.35

    for ax, matrix, title in [(axes[0], dice_mat, "Dice Score"),
                               (axes[1], iou_mat, "IoU")]:
        for j, cls in enumerate(classes):
            offset = (j - (n_cls - 1) / 2) * width
            bars = ax.bar(x + offset, matrix[:, j], width,
                         label=FG_LABELS[cls], color=PALETTE[j], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=9, rotation=15, ha="right")
        ymin = max(0, matrix[matrix > 0].min() - 0.05) if (matrix > 0).any() else 0
        ax.set_ylim(ymin, 1.02)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison — Dataset501_MickeyScroll", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, "model_comparison.png", save)


def plot_foreground_summary(experiments: dict[str, dict], save: bool = False):
    """Single bar chart: foreground-mean Dice for each model (quick overview)."""
    names = list(experiments.keys())
    dices = [_fg_mean(experiments[n]).get("Dice", 0) for n in names]
    ious = [_fg_mean(experiments[n]).get("IoU", 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(names)), 5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w / 2, dices, w, label="Dice", color=PALETTE[0], alpha=0.85)
    b2 = ax.bar(x + w / 2, ious, w, label="IoU", color=PALETTE[1], alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ymin = max(0, min(dices + ious) - 0.05) if dices else 0
    ax.set_ylim(ymin, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Foreground-Mean Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, "foreground_summary.png", save)


def plot_per_case(experiments: dict[str, dict], save: bool = False):
    """Per-case Dice box plots side-by-side for every model."""
    n_exp = len(experiments)
    if n_exp == 0:
        return

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5), squeeze=False)
    axes = axes[0]

    for i, (exp_name, fold_paths) in enumerate(experiments.items()):
        per_class = {str(cls): [] for cls in FG_LABELS}
        for path in fold_paths.values():
            summary = load_summary(path)
            for case in extract_per_case_metrics(summary):
                metrics = case.get("metrics", {})
                for cls in per_class:
                    d = metrics.get(cls, {}).get("Dice")
                    if d is not None:
                        per_class[cls].append(d)

        data = [per_class[str(cls)] for cls in sorted(FG_LABELS.keys())]
        labels = [FG_LABELS[cls] for cls in sorted(FG_LABELS.keys())]

        if any(len(d) > 0 for d in data):
            bp = axes[i].boxplot(data, labels=labels, patch_artist=True,
                                medianprops=dict(color="black"))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        axes[i].set_title(exp_name, fontsize=10)
        axes[i].set_ylabel("Dice" if i == 0 else "")
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Case Dice Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, "per_case_comparison.png", save)


def plot_training_curves(experiments: dict[str, dict], save: bool = False):
    """Overlay training-loss curves for MONAI models (from training_log.json)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found = False

    for i, (exp_name, fold_paths) in enumerate(experiments.items()):
        for fold_name, summary_path in fold_paths.items():
            log_path = summary_path.parent.parent / "training_log.json"
            if not log_path.exists():
                continue
            log = json.loads(log_path.read_text())
            losses = log.get("train_loss", [])
            if not losses:
                continue
            ax.plot(losses, label=f"{exp_name} ({fold_name})",
                    color=PALETTE[i % len(PALETTE)], alpha=0.8)
            found = True

    if not found:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Curves", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, "training_curves.png", save)


def _save_or_show(fig, filename: str, save: bool):
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    else:
        plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all model results")
    parser.add_argument("--save", action="store_true", help="Save figures instead of showing")
    args = parser.parse_args()

    experiments = discover_experiments()
    if not experiments:
        print("No experiments found. Checked:")
        for ds in DATASET_NAMES:
            print(f"  nnU-Net:  {NNUNET_RESULTS_BASE / ds}")
            print(f"  MONAI:    {MONAI_RESULTS_BASE / ds}")
        print("Train at least one model first.")
        exit(1)

    print(f"Found {len(experiments)} experiment(s):")
    for name, folds in experiments.items():
        print(f"  - {name}  ({len(folds)} fold(s))")

    print_comparison_table(experiments)
    plot_foreground_summary(experiments, save=args.save)
    plot_comparison(experiments, save=args.save)
    plot_per_case(experiments, save=args.save)
    plot_training_curves(experiments, save=args.save)
