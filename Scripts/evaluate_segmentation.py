"""Score nnU-Net validation NIfTIs against the supplied manual labels.

The aggregate matches Scripts/fair_compare.py: each foreground class (film base
and emulsion) in each held-out case contributes one value to the macro mean.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


METRICS = ("Dice", "IoU", "Precision", "Recall")


def score_case(pred_path: Path, label_path: Path) -> dict:
    pred = np.asarray(nib.load(str(pred_path)).dataobj, dtype=np.uint8)
    label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)
    if pred.shape != label.shape:
        raise ValueError(f"Shape mismatch for {pred_path.name}: {pred.shape} vs {label.shape}")

    result = {}
    for cls in (1, 2):
        tp = fp = fn = 0
        for p_slice, l_slice in zip(pred, label):
            p = p_slice == cls
            l = l_slice == cls
            tp += int(np.count_nonzero(p & l))
            fp += int(np.count_nonzero(p & ~l))
            fn += int(np.count_nonzero(~p & l))
        denom = 2 * tp + fp + fn
        result[str(cls)] = {
            "Dice": 2 * tp / denom if denom else float("nan"),
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "Precision": tp / (tp + fp) if tp + fp else float("nan"),
            "Recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = sorted(args.pred_dir.glob("*.nii.gz"))
    if not predictions:
        parser.error(f"No .nii.gz predictions in {args.pred_dir}")
    per_case = {}
    for pred_path in predictions:
        case = pred_path.name.removesuffix(".nii.gz")
        label_path = args.labels_dir / f"{case}.nii.gz"
        if not label_path.is_file():
            parser.error(f"Missing label: {label_path}")
        per_case[case] = score_case(pred_path, label_path)
        print(f"{case}: Dice base={per_case[case]['1']['Dice']:.4f}, "
              f"emulsion={per_case[case]['2']['Dice']:.4f}", flush=True)

    aggregate = {
        metric: float(np.nanmean([
            per_case[case][str(cls)][metric]
            for case in per_case for cls in (1, 2)
        ]))
        for metric in METRICS
    }
    result = {"per_case": per_case, "aggregate": aggregate,
              "n_cases": len(per_case), "averaging": "case x foreground class macro mean"}
    print(json.dumps(aggregate, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
