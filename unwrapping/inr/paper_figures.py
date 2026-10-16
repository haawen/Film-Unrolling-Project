"""Build paper-ready 3-panel figures: GT | Oracle | R1n.

For each HQ outer preset (clean, imperfect), produces:
  - one single-frame panel (256 x ~455 px per panel)  -- frame.png
  - one two-frame window  (256 x ~910 px per panel)   -- window.png

The "oracle" is the strip rendered by sampling the CT volume at the GT u_map
(render_oracle_strip.py): no geometric error, only CT noise / binning error.
The "R1n" is our self-supervised inverse-mapping INR.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _resample_row_nearest(row, new_len):
    old_len = row.size
    if old_len == new_len:
        return row.copy()
    idx = np.minimum((np.arange(new_len) * old_len / new_len).astype(np.int64),
                     old_len - 1)
    return row[idx]


def resample_strip(strip, new_W):
    if strip.shape[1] == new_W:
        return strip.copy()
    out = np.empty((strip.shape[0], new_W), dtype=strip.dtype)
    for z in range(strip.shape[0]):
        out[z] = _resample_row_nearest(strip[z], new_W)
    return out


def _circ_shift_align(a, b):
    """Circular-shift align `a` onto `b` using row-mean cross correlation."""
    fa = np.fft.rfft(a.mean(axis=0))
    fb = np.fft.rfft(b.mean(axis=0))
    xc = np.fft.irfft(np.conj(fa) * fb, n=a.shape[1])
    shift = int(np.argmax(xc))
    return np.roll(a, shift, axis=1), shift


def normalize_per_row(strip, lo=2.0, hi=98.0):
    """Per-row percentile clip + linear stretch to [0, 1]."""
    out = np.empty_like(strip, dtype=np.float32)
    for z in range(strip.shape[0]):
        a, b = np.percentile(strip[z], [lo, hi])
        if b - a < 1e-6:
            out[z] = 0.0
        else:
            out[z] = np.clip((strip[z] - a) / (b - a), 0.0, 1.0)
    return out


def pick_best_content_window(gt_strip, win_w, stride=None, frame_w=None):
    """Find column c0 in gt_strip maximizing std over (c0, c0+win_w).

    If frame_w is given, c0 is constrained to multiples of frame_w so the
    crop starts on a movie-frame boundary.
    """
    H, W = gt_strip.shape
    stride = stride or max(1, win_w // 8)
    if frame_w is not None:
        # Snap stride to frame boundaries.
        stride = frame_w
    best_c0, best_std = 0, -1.0
    for c0 in range(0, W - win_w + 1, stride):
        v = float(gt_strip[:, c0:c0 + win_w].std())
        if v > best_std:
            best_std, best_c0 = v, c0
    return best_c0


def make_3panel(gt, oracle, ours, c0, c1, title_suffix, out_path,
                ours_label="R1n (SS)"):
    n_rows = gt.shape[0]
    pw = (c1 - c0) / 100.0
    ph = n_rows / 100.0
    fig, axes = plt.subplots(1, 3,
                              figsize=(pw * 3 + 0.3, ph + 0.55),
                              constrained_layout=True)
    panels = [(gt, "Ground truth (source)"),
              (oracle, "Oracle (CT sampled at GT u-map)"),
              (ours, ours_label)]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(np.clip(img[:, c0:c1], 0, 1), cmap="gray", aspect="equal",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title_suffix, fontsize=9)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def process_preset(preset, gt_path, oracle_path, ours_path, out_dir,
                    ours_label="R1n (SS)"):
    print(f"\n=== {preset} ===")
    gt = np.load(gt_path)["strip"].astype(np.float32)
    oracle = np.load(oracle_path)["strip"].astype(np.float32)
    ours = np.load(ours_path)["strip"].astype(np.float32)
    print(f"  GT     {gt.shape} [{gt.min():.3g}, {gt.max():.3g}]")
    print(f"  oracle {oracle.shape} [{oracle.min():.3g}, {oracle.max():.3g}]")
    print(f"  ours   {ours.shape} [{ours.min():.3g}, {ours.max():.3g}]")

    # Resample to GT resolution (one source frame = ~455 cols). The model
    # rendered at lower resolution; nearest-neighbor up-sample keeps one
    # frame at its natural pixel size in every panel.
    W_ref = gt.shape[1]
    gt_r = gt
    oracle_r = oracle if oracle.shape[1] == W_ref else resample_strip(oracle, W_ref)
    ours = resample_strip(ours, W_ref)

    # Normalize for visual comparison.
    gt_v = normalize_per_row(gt_r)
    oracle_v = normalize_per_row(oracle_r)
    ours_v = normalize_per_row(ours)

    # Align oracle and ours to gt by circular shift on row-mean
    oracle_v, soff = _circ_shift_align(oracle_v, gt_v)
    ours_v, sour = _circ_shift_align(ours_v, gt_v)
    print(f"  align shifts: oracle={soff} px, ours={sour} px")

    # Pick best-content positions in GT. The synthetic generator lays movie
    # frames at 4:3 aspect (frame_w = h * 4 // 3), tiled end-to-end with no
    # gap. Snap the crop start to a multiple of frame_w so panels show
    # whole frames instead of straddling a frame boundary.
    n_rows = gt_v.shape[0]
    frame_w = (n_rows * 4) // 3   # 341 for h=256, matches generate.py
    win_w = frame_w * 2

    c0_f = pick_best_content_window(gt_v, frame_w, frame_w=frame_w)
    c1_f = c0_f + frame_w

    c0_w = pick_best_content_window(gt_v, win_w, frame_w=frame_w)
    c1_w = c0_w + win_w

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    suffix = preset.replace("hq_video_4k_", "")
    make_3panel(gt_v, oracle_v, ours_v, c0_f, c1_f,
                f"{preset} -- single frame  (cols {c0_f}-{c1_f})",
                os.path.join(out_dir, f"3panel_{suffix}_frame.png"),
                ours_label=ours_label)
    make_3panel(gt_v, oracle_v, ours_v, c0_w, c1_w,
                f"{preset} -- 2-frame window  (cols {c0_w}-{c1_w})",
                os.path.join(out_dir, f"3panel_{suffix}_window.png"),
                ours_label=ours_label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines-dir", default="baselines_hq_outer")
    p.add_argument("--out-dir", default="docs/figures")
    args = p.parse_args()

    presets = [
        ("hq_video_4k_outer",
         "gt_clean.npz", "oracle_clean.npz", "R1n_clean/strip.npz"),
        ("hq_video_4k_outer_imperfect",
         "gt_imperfect.npz", "oracle_imperfect.npz", "R1n_imperfect/strip.npz"),
    ]
    bd = args.baselines_dir
    for name, gt, oracle, ours in presets:
        process_preset(name,
                       os.path.join(bd, gt),
                       os.path.join(bd, oracle),
                       os.path.join(bd, ours),
                       args.out_dir)


if __name__ == "__main__":
    main()
