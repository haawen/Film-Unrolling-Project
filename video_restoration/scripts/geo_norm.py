"""Geometry-aware radiometric normalization PROTOTYPE (video-positive).

strip rows = CT z-slices (per-slice gain), strip cols = arc (windings concatenated).
Illumination modelled as  row_gain[z] * arc_trend[arc]  estimated from CONTENT-MASKED
background only (so it does NOT follow Mickey -> no halos, unlike a blind low-pass).

Outputs per crop:  original | estimated illumination field | corrected
"""
import numpy as np
from scipy import ndimage
from PIL import Image

STRIP = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
OUT   = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad"

mm = np.load(STRIP, mmap_mode='r')
H, W = mm.shape
cols_s = np.arange(2000, W-2000, 40)                    # subsample for global stats
samp = np.asarray(mm[:, cols_s], np.float32)
lo, hi = np.percentile(samp, [1, 99])
to_v = lambda s: 1.0 - np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)   # --invert

# ---- content mask: Mickey is DARK in video (<0.40); highlights very bright (>0.97) ----
def bg_mask(v):
    content = (v < 0.40) | (v > 0.97)
    content = ndimage.binary_dilation(content, iterations=4)
    return ~content

# ---- per-slice (row) gain from the FULL strip, background only ----
vs = to_v(samp)
mask = bg_mask(vs)
vm = np.where(mask, vs, np.nan)
row_gain = np.nanmedian(vm, axis=1)                     # (H,)
row_gain = np.where(np.isfinite(row_gain), row_gain, np.nanmedian(row_gain))
row_gain = ndimage.gaussian_filter1d(row_gain, 2)       # keep a smooth gain curve
g0 = np.median(row_gain)
print(f"row-gain (per-slice level): std {row_gain.std():.4f} range {row_gain.min():.3f}..{row_gain.max():.3f}")

g8 = lambda a: np.repeat((np.clip(a,0,1)*255).astype(np.uint8)[...,None],3,2)

for name, x0 in [("mid_b", 120000), ("late", 170000)]:
    v = to_v(np.asarray(mm[:, x0:x0+1150], np.float32))
    # 1) per-slice normalization (divide each row by its gain)
    v1 = v / (row_gain[:, None] / g0 + 1e-3)
    # 2) smooth arc trend from row-normalized background (heavy smoothing along arc)
    m = bg_mask(v1)
    col_bg = np.where(m, v1, np.nan)
    arc = np.nanmedian(col_bg, axis=0)                  # (1150,)
    arc = np.where(np.isfinite(arc), arc, np.nanmedian(arc))
    arc = ndimage.gaussian_filter1d(arc, 60)
    a0 = np.median(arc)
    v2 = np.clip(v1 / (arc[None, :] / a0 + 1e-3), 0, 1)
    # 3) non-separable broad 2D residual from BACKGROUND only (normalized convolution:
    #    smooth over known bg pixels, content excluded -> no halos), then divide.
    m2 = bg_mask(v2).astype(np.float32)
    s = 90.0
    R = ndimage.gaussian_filter(v2*m2, s) / (ndimage.gaussian_filter(m2, s) + 1e-6)
    r0 = np.median(R)
    v3 = np.clip(v2 / (R / r0 + 1e-3), 0, 1)

    # total illumination field removed (for display), stretched
    field = (row_gain[:, None]/g0) * (arc[None, :]/a0) * (R/r0)
    fv = (field - field.min()) / (field.max() - field.min() + 1e-6)

    gap = np.full((H, 6, 3), 30, np.uint8)
    Image.fromarray(np.concatenate([g8(v), gap, g8(fv), gap, g8(v3)], 1)).save(f"{OUT}/geo_{name}.png")
    rb = np.nanstd(np.nanmedian(np.where(bg_mask(v),  v,  np.nan), axis=1))
    ra = np.nanstd(np.nanmedian(np.where(bg_mask(v3), v3, np.nan), axis=1))
    print(f"{name}: bg row-level std {rb:.4f} -> {ra:.4f}; bg 2D std {np.nanstd(np.where(bg_mask(v),v,np.nan)):.4f} -> {np.nanstd(np.where(bg_mask(v3),v3,np.nan)):.4f}")
print("saved")
