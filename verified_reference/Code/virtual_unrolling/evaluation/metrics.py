"""Similarity / quality metrics for the film video comparison.

All "classical" metrics here are self-contained (numpy/scipy) so they run in the
local thesis env (no skimage/cv2 needed). The learned perceptual metrics
(DISTS/LPIPS) are gated behind an optional `pyiqa` import — `HAS_PYIQA` is
False when it is unavailable, and the wrappers raise a clear error.

Conventions: images are float32 2D arrays in [0, 1]. "Higher is better" unless
noted. Cross-modal pairs should be histogram-matched before the intensity-based
metrics (SSIM/PSNR); NMI and gradient-correlation tolerate the modality gap.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

try:  # learned metrics (optional)
    import torch
    import pyiqa
    HAS_PYIQA = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PYIQA = False


# --------------------------------------------------------------------------- #
# Cross-modal / alignment-robust (primary)
# --------------------------------------------------------------------------- #
def normalized_mutual_information(a, b, bins=64):
    """Normalized mutual information (Studholme), range ~[1, 2]; higher = better.

    The standard cross-modal similarity: makes no assumption about the intensity
    relationship between modalities (CT-derived vs optical), so it survives the
    negative/positive + contrast differences that wreck PSNR/SSIM.
    """
    a = a.ravel()
    b = b.ravel()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / max(hist.sum(), 1e-12)
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)

    def _entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h_a, h_b = _entropy(pa), _entropy(pb)
    h_ab = _entropy(pab.ravel())
    if h_ab <= 0:
        return float("nan")
    return (h_a + h_b) / h_ab


def gradient_correlation(a, b):
    """Pearson correlation of gradient-magnitude maps; range [-1, 1].

    Edge structure is modality-invariant, so this measures whether the same
    contours appear in the same places without caring about absolute intensity.
    """
    ga = np.hypot(sobel(a, axis=0), sobel(a, axis=1))
    gb = np.hypot(sobel(b, axis=0), sobel(b, axis=1))
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    denom = np.sqrt((ga * ga).sum() * (gb * gb).sum())
    if denom <= 0:
        return float("nan")
    return float((ga * gb).sum() / denom)


# --------------------------------------------------------------------------- #
# Structural (secondary; use on histogram-matched, registered frames)
# --------------------------------------------------------------------------- #
def _ssim_map(a, b, sigma=1.5, L=1.0):
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = gaussian_filter(a * a, sigma) - mu_a2
    vb = gaussian_filter(b * b, sigma) - mu_b2
    vab = gaussian_filter(a * b, sigma) - mu_ab
    num = (2 * mu_ab + c1) * (2 * vab + c2)
    den = (mu_a2 + mu_b2 + c1) * (va + vb + c2)
    return num / np.clip(den, 1e-12, None)


def ssim(a, b, sigma=1.5):
    """Mean SSIM, range [-1, 1]; higher = better."""
    return float(_ssim_map(a, b, sigma).mean())


def ms_ssim(a, b, scales=4, sigma=1.5):
    """Multi-scale SSIM (mean of contrast-structure across scales x luminance).

    A lightweight MS-SSIM: SSIM at successively half-sized images, geometric
    mean. Robust-ish to the resolution mismatch once frames are resized common.
    """
    vals = []
    ca, cb = a, b
    for s in range(scales):
        vals.append(ssim(ca, cb, sigma))
        if s < scales - 1:
            ca = gaussian_filter(ca, sigma)[::2, ::2]
            cb = gaussian_filter(cb, sigma)[::2, ::2]
            if min(ca.shape) < 8:
                break
    vals = np.clip(vals, 1e-6, None)
    return float(np.exp(np.mean(np.log(vals))))


def psnr(a, b, L=1.0):
    """Peak SNR (dB). Intensity-sensitive — report only post histogram-match."""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10 * np.log10(L * L / mse))


def msad(a, b):
    """Mean sum of absolute differences (ffmpeg-quality-metrics' MSAD), on
    [0,1]. Lower=better. Intensity-sensitive (like PSNR) — secondary."""
    return float(np.mean(np.abs(a - b)))


# --------------------------------------------------------------------------- #
# Learned perceptual + no-reference (optional, via pyiqa)
# --------------------------------------------------------------------------- #
_PYIQA_CACHE = {}


def _pyiqa_metric(name, device="cpu"):
    if not HAS_PYIQA:
        raise RuntimeError(
            f"pyiqa not available; cannot compute '{name}'. "
            "pip install pyiqa (run on Merlin7) to enable learned/NR metrics."
        )
    key = (name, device)
    if key not in _PYIQA_CACHE:
        _PYIQA_CACHE[key] = pyiqa.create_metric(name, device=device)
    return _PYIQA_CACHE[key]


def _to_tensor(img):
    """2D [0,1] float -> (1,3,H,W) tensor (replicate gray to 3 channels)."""
    t = torch.from_numpy(np.ascontiguousarray(img)).float()[None, None]
    return t.repeat(1, 3, 1, 1)


def dists(a, b, device="cpu"):
    """DISTS (full-reference, structure+texture; misalignment-tolerant). Lower=better."""
    m = _pyiqa_metric("dists", device)
    return float(m(_to_tensor(a), _to_tensor(b)).item())


def lpips(a, b, device="cpu"):
    """LPIPS (full-reference perceptual). Lower=better. Needs aligned frames."""
    m = _pyiqa_metric("lpips", device)
    return float(m(_to_tensor(a), _to_tensor(b)).item())


def fr_pair_metrics(a, b):
    """All classical full-reference metrics for one aligned, histogram-matched pair
    (LPIPS/DISTS are computed separately by the caller)."""
    out = {
        "nmi": normalized_mutual_information(a, b),
        "grad_corr": gradient_correlation(a, b),
        "ssim": ssim(a, b),
        "ms_ssim": ms_ssim(a, b),
        "psnr": psnr(a, b),
        "msad": msad(a, b),
    }
    return out
