"""
Unsupervised scoring for unwrapped film strips.

Real Mickey has no ground truth u-map, so row_corr_mean isn't available.
This module computes proxy metrics that should correlate with unwrapping
quality — validated on synthetic where we can compare against row_corr.

Metrics (all higher = better except where noted):

  z_coherence       Adjacent z-rows of a correctly-unwrapped strip capture
                    the same film emulsion, so they should be nearly
                    identical. Mean Pearson correlation of (row_z, row_z+1)
                    pairs. Range [-1, 1]. Target: >0.95 for clean.

  hf_energy_ratio   Fraction of horizontal power in the top 25% of
                    frequencies. Correct unwrapping preserves sharp
                    edges (perforations, frame borders). Blurred /
                    misaligned unwrapping attenuates HF content.
                    Range [0, 1]. Target: ~0.1-0.3 for sharp strips.

  dynamic_range     p95 - p5 of strip intensities. A correct strip has
                    bright emulsion features against dark background.
                    A collapsed/averaged strip compresses this range.
                    Range: data-dependent (typically [0, 1]).

  row_var_mean      Mean per-row variance. A strip showing varied
                    content (not a flat band) has higher variance.
                    Sanity check for collapse.

  emul_fraction     Fraction of non-NaN (or non-zero) pixels.
                    Coverage check — a correctly-fit surface covers
                    the full (u, z) grid.

Usage (CLI):
    python -m unwrapping.inr.strip_score <strip.npz> [--out-dir DIR]
"""

import argparse
import json
import os

import numpy as np


def _clean_strip(strip):
    """Return (valid_mask, strip_zeroed). NaNs replaced with row mean."""
    valid = np.isfinite(strip)
    # If NaN-free, treat zeros as background (keep as-is)
    if not np.all(valid):
        cleaned = strip.copy()
        for z in range(strip.shape[0]):
            row = strip[z]
            m = np.isfinite(row)
            if m.any():
                cleaned[z, ~m] = row[m].mean()
            else:
                cleaned[z] = 0.0
        return valid, cleaned
    return valid, strip.astype(np.float32)


def z_coherence(strip):
    """Mean Pearson correlation between adjacent z-rows.

    For a correctly unwrapped strip, adjacent z-slices produce nearly
    identical rendered rows — the film is a z-invariant object locally.
    Misalignment or collapse breaks this.
    """
    if strip.shape[0] < 2:
        return float("nan")
    _, s = _clean_strip(strip)
    rows = s
    # Pearson correlation of (z, z+1) pairs
    corrs = []
    for z in range(rows.shape[0] - 1):
        a, b = rows[z], rows[z + 1]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom > 0:
            corrs.append(float((a * b).sum() / denom))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def hf_energy_ratio(strip, hf_fraction=0.25):
    """Fraction of 1D horizontal power in the top `hf_fraction` frequencies.

    Averaged across rows. Captures whether the strip has sharp horizontal
    features (edges, perforations) vs being horizontally smooth.
    """
    _, s = _clean_strip(strip)
    n = s.shape[1]
    # remove per-row mean (DC) before FFT
    s = s - s.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(s, axis=1)
    power = np.abs(spec) ** 2  # (Z, n_freqs)
    total = power.sum(axis=1) + 1e-12
    k_cut = int(power.shape[1] * (1.0 - hf_fraction))
    hf = power[:, k_cut:].sum(axis=1)
    return float((hf / total).mean())


def dynamic_range(strip, p_lo=5, p_hi=95):
    _, s = _clean_strip(strip)
    lo = np.percentile(s, p_lo)
    hi = np.percentile(s, p_hi)
    return float(hi - lo)


def row_var_mean(strip):
    _, s = _clean_strip(strip)
    return float(s.var(axis=1).mean())


def emul_fraction(strip):
    if np.issubdtype(strip.dtype, np.floating):
        valid = np.isfinite(strip) & (strip != 0)
    else:
        valid = strip != 0
    return float(valid.mean())


def score_strip(strip):
    """Run all unsupervised metrics on a strip. Returns dict."""
    return {
        "shape": list(strip.shape),
        "z_coherence": z_coherence(strip),
        "hf_energy_ratio": hf_energy_ratio(strip),
        "dynamic_range": dynamic_range(strip),
        "row_var_mean": row_var_mean(strip),
        "emul_fraction": emul_fraction(strip),
    }


def load_strip(path):
    """Load strip from .npz (key='strip') or raw .npy."""
    if path.endswith(".npz"):
        d = np.load(path)
        if "strip" in d.files:
            return d["strip"].astype(np.float32)
        raise ValueError(f"{path} has no 'strip' key; found {d.files}")
    return np.load(path).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("strip", help="Path to strip.npz (or .npy)")
    p.add_argument("--out-dir", default=None,
                   help="If set, write scores.json here")
    p.add_argument("--tag", default=None,
                   help="Optional tag echoed into output")
    args = p.parse_args()

    strip = load_strip(args.strip)
    scores = score_strip(strip)
    if args.tag:
        scores["tag"] = args.tag
    scores["source"] = os.path.abspath(args.strip)

    print(json.dumps(scores, indent=2))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "unsup_scores.json"), "w") as f:
            json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
