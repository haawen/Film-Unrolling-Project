"""Video I/O and frame helpers shared by the evaluation tools.

Verbatim copies of the helpers that frame_match, stabilize_affine,
evaluate_final_vs_gt and gt_perflock use (formerly in compare_videos.py and
stabilize_horizontal.py, whose command-line tools are not part of the pipeline).
"""
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import sobel


def load_video_luma(path):
    """Read an mp4 -> (N, H, W) float32 luma in [0, 1]."""
    r = imageio.get_reader(path)
    frames = []
    for f in r:
        f = np.asarray(f, dtype=np.float32)
        if f.ndim == 3:  # RGB -> Rec.601 luma
            f = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        frames.append(f)
    r.close()
    arr = np.stack(frames)
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr


def trim_leader(frames, diff_thresh=1.0 / 255):
    """Drop leading near-duplicate frames (static leader/title before first cut).

    Returns (trimmed_frames, n_dropped). Stops at the first frame whose mean
    abs-difference from its predecessor exceeds the threshold.
    """
    n = 0
    for i in range(1, len(frames)):
        if np.abs(frames[i] - frames[i - 1]).mean() > diff_thresh:
            n = i
            break
    return frames[n:], n


def parse_crop(s):
    if not s:
        return None
    return tuple(float(x) for x in s.split(","))


def apply_crop(frames, box):
    if box is None:
        return frames
    _, h, w = frames.shape
    x0, y0, x1, y1 = box
    return frames[:, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def resize_to_height(frames, height):
    out = []
    for f in frames:
        w = max(2, int(round(height * f.shape[1] / f.shape[0])))
        im = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))
        out.append(np.asarray(im.resize((w, height))).astype(np.float32) / 255)
    return out


def histogram_match(src, ref, bins=256):
    """Match src's intensity histogram to ref's (per-pair, for SSIM/PSNR)."""
    s = src.ravel()
    sv, idx, cnt = np.unique(s, return_inverse=True, return_counts=True)
    s_cdf = np.cumsum(cnt).astype(np.float64) / s.size
    rv, rc = np.unique(ref.ravel(), return_counts=True)
    r_cdf = np.cumsum(rc).astype(np.float64) / ref.size
    interp = np.interp(s_cdf, r_cdf, rv)
    return interp[idx].reshape(src.shape).astype(np.float32)


def grad_mag(img):
    """Normalized Sobel gradient magnitude (modality-invariant edge map)."""
    m = np.hypot(sobel(img, axis=0), sobel(img, axis=1))
    m -= m.min()
    mx = m.max()
    return (m / mx).astype(np.float32) if mx > 0 else m


def common_crop(a, b):
    """Crop two same-height arrays to common width (left-aligned)."""
    w = min(a.shape[1], b.shape[1])
    return a[:, :w], b[:, :w]


def aggregate(per_pair):
    keys = {k for d in per_pair for k in d}
    agg = {}
    for k in keys:
        vals = [d[k] for d in per_pair if k in d and np.isfinite(d[k])]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
    return agg


def save_montage(triplets, out_path, n=6):
    """triplets: list of (pred, gt, diff). Save n evenly-spaced rows."""
    if not triplets:
        return
    idx = np.linspace(0, len(triplets) - 1, min(n, len(triplets)), dtype=int)
    rows = []
    for k in idx:
        p, g, d = triplets[k]
        h = p.shape[0]
        gap = np.ones((h, 4), np.float32)
        rows.append(np.concatenate([p, gap, g, gap, d], axis=1))
    H = max(r.shape[0] for r in rows)
    Wm = max(r.shape[1] for r in rows)
    canvas = np.ones((H * len(rows) + 4 * len(rows), Wm), np.float32)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 4
    Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).save(out_path)


def load_luma(path):
    r = imageio.get_reader(path)
    fr = []
    for f in r:
        a = np.asarray(f, np.float32)
        if a.ndim == 3:
            a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        fr.append(a / 255.0)
    r.close()
    return fr


def gmag(f):
    m = np.hypot(sobel(f, 0), sobel(f, 1))
    mx = m.max()
    return m / mx if mx > 0 else m


def _resize(f, h, w):
    return np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)
                      ).resize((w, h))).astype(np.float32) / 255


def phase2d(a, b, lim=0.2):
    """Sub-pixel (dy, dx) aligning b onto a (2-D FFT phase corr, bounded).
    Full-frame -> dominated by the static film structure, not the character."""
    A = np.fft.rfft2(a - a.mean()); B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B); R /= np.abs(R) + 1e-12
    r = np.fft.irfft2(R, s=a.shape)
    h, w = a.shape
    my, mx = max(1, int(lim * h)), max(1, int(lim * w))
    rows = np.zeros(h, bool); rows[:my + 1] = rows[-my:] = True
    cols = np.zeros(w, bool); cols[:mx + 1] = cols[-mx:] = True
    r = np.where(rows[:, None] & cols[None, :], r, -np.inf)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    def sub(a0, a1, a2):
        if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(a2)):
            return 0.0
        d = a0 - 2 * a1 + a2
        return 0.5 * (a0 - a2) / d if abs(d) > 1e-9 else 0.0
    fy = sub(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    fx = sub(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = (py if py <= h // 2 else py - h) + fy
    dx = (px if px <= w // 2 else px - w) + fx
    return dy, dx
