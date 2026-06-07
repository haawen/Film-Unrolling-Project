"""Render mid-roll CT slice + segmentation overlay PNGs for human review.

Picks N evenly spaced z-positions across the whole roll, loads CT + softmax
(argmax → label), saves three panels each: CT only, label only, overlay.
"""
import argparse
import os
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_chunk_pairs(pairs_dir):
    pairs = {}
    for f in sorted(os.listdir(pairs_dir)):
        m = re.match(r"volume_(\d+)-(\d+)\.h5$", f)
        if not m:
            continue
        z0, z1 = int(m.group(1)), int(m.group(2))
        prob = f.replace(".h5", "_Probabilities.h5")
        if not os.path.exists(os.path.join(pairs_dir, prob)):
            continue
        pairs[(z0, z1)] = (f, prob)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=8)
    args = p.parse_args()

    pairs = collect_chunk_pairs(args.pairs_dir)
    if not pairs:
        raise SystemExit(f"No paired chunks in {args.pairs_dir}")
    # Sample within the actual chunk coverage, not [0, max_z]. This matters
    # for smoke runs that have only a single mid-roll chunk.
    z_min = min(z0 for (z0, _) in pairs)
    z_max = max(z1 for (_, z1) in pairs)
    if z_min == z_max:
        targets = [z_min]
    else:
        targets = [int(round(z_min + (z_max - z_min) * (i + 1) / (args.n_samples + 1)))
                   for i in range(args.n_samples)]
    print(f"Chunk z-range = [{z_min}, {z_max}].  Sampling z indices: {targets}")
    os.makedirs(args.out_dir, exist_ok=True)

    for tz in targets:
        chunk = next(((z0, z1) for (z0, z1) in pairs if z0 <= tz <= z1), None)
        if chunk is None:
            print(f"  z={tz}: no chunk; skip"); continue
        z0, z1 = chunk
        vf, pf = pairs[chunk]
        local = tz - z0
        with h5py.File(os.path.join(args.pairs_dir, vf), "r") as h:
            ct = h["volume"][local].astype(np.float32)
        with h5py.File(os.path.join(args.pairs_dir, pf), "r") as h:
            probs = h["exported_data"][local]      # (H, W, C)
        label = np.argmax(probs, axis=-1).astype(np.uint8)

        # Normalise CT to [0,1] using slice percentiles
        lo, hi = np.percentile(ct, [1, 99])
        ct_n = np.clip((ct - lo) / max(hi - lo, 1e-6), 0, 1)

        # Three-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        axes[0].imshow(ct_n, cmap="gray", interpolation="nearest")
        axes[0].set_title(f"CT z={tz}")
        # Label as RGB: 0=black, 1=cyan, 2=magenta
        rgb = np.zeros((*label.shape, 3), dtype=np.float32)
        rgb[label == 1] = [0.0, 0.8, 0.9]
        rgb[label == 2] = [0.9, 0.2, 0.6]
        axes[1].imshow(rgb, interpolation="nearest")
        axes[1].set_title("Label (1=cyan film_base, 2=magenta emul)")
        # Overlay
        overlay = np.stack([ct_n, ct_n, ct_n], axis=-1) * 0.7 + rgb * 0.5
        axes[2].imshow(np.clip(overlay, 0, 1), interpolation="nearest")
        axes[2].set_title("Overlay")
        for a in axes:
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"sample_z{tz:04d}.png")
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  wrote {out}")

    print(f"Done. {len(targets)} sample PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
