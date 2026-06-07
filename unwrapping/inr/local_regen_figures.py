"""Regenerate frame_compare.png / zoom_window_*.png / gt_compare.png for the
HQ outer eval cells, locally, from strip.npz + ground_truth.npz.

Runs in seconds — no model, no volume, no SLURM round-trip. Drops the new
figures next to the existing eval artefacts under baselines_hq_outer/.
"""
import os
import sys
import numpy as np

# Import the updated figure functions from surface_eval.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from unwrapping.inr.surface_eval import (
    save_strip_figures,
    save_zoomed_strip_windows,
    save_single_frame_compare,
    _resample_row_nearest,
)


def main():
    cells = [
        # (label, strip_npz, gt_npz)
        ("R1S_clean",     "baselines_hq_outer/R1S_clean/strip.npz",
                          "baselines_hq_outer/gt_clean.npz"),
        ("R1n_clean",     "baselines_hq_outer/R1n_clean/strip.npz",
                          "baselines_hq_outer/gt_clean.npz"),
        ("R1S_imperfect", "baselines_hq_outer/R1S_imperfect/strip.npz",
                          "baselines_hq_outer/gt_imperfect.npz"),
        ("R1n_imperfect", "baselines_hq_outer/R1n_imperfect/strip.npz",
                          "baselines_hq_outer/gt_imperfect.npz"),
    ]
    for label, strip_path, gt_path in cells:
        out_dir = os.path.dirname(strip_path)
        print(f"\n=== {label} -> {out_dir} ===")
        s = np.load(strip_path)
        strip = s["strip"]
        n_layers = int(s["n_layers"])
        ppw = int(s["pixels_per_winding"])
        gt = np.load(gt_path)
        gt_strip = gt["strip"]

        # Align prediction to GT width with nearest-neighbor (matches
        # surface_eval.evaluate_against_gt).
        H_p, W_p = strip.shape
        H_g, W_g = gt_strip.shape
        if W_p != W_g:
            print(f"  resampling pred {strip.shape} -> ({H_p}, {W_g})")
            aligned = np.empty((H_p, W_g), dtype=strip.dtype)
            for z in range(H_p):
                aligned[z] = _resample_row_nearest(strip[z], W_g)
        else:
            aligned = strip
        # Circular-shift register prediction onto GT (using row-mean xcorr).
        fa = np.fft.rfft(aligned.mean(axis=0))
        fb = np.fft.rfft(gt_strip.mean(axis=0))
        xc = np.fft.irfft(np.conj(fa) * fb, n=aligned.shape[1])
        shift = int(np.argmax(xc))
        aligned = np.roll(aligned, shift, axis=1)
        print(f"  align shift = {shift} px")

        # GT crop to same width (already same here)
        gt_crop = gt_strip
        # Cap any out-of-range values from interpolation
        aligned = np.clip(aligned, 0.0, 1.0)
        gt_crop = np.clip(gt_crop, 0.0, 1.0)

        # Regenerate the figures with the frame-snapped logic.
        save_zoomed_strip_windows(aligned, gt_crop, out_dir)
        save_single_frame_compare(aligned, gt_crop, out_dir)
        # Also regen the "detail_layers_*" tile (uses frame_w internally? no —
        # this one is winding-based, unrelated to frame boundaries; skip.)
        print(f"  wrote frame_compare.png + zoom_window_*.png")


if __name__ == "__main__":
    main()
