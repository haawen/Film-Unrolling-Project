"""Geometry-aware normalization v2 (tuning): single ANISOTROPIC content-masked
illumination field  F = normconv(v; sigma_z small, sigma_arc large)  -> captures
per-slice (z) banding AND smooth along-film (arc) shading in one faithful field.
Estimated over a WIDER context than the displayed crop (no edge bias). Tunable
field-scale (sigma_arc) and correction strength (alpha)."""
import numpy as np
from scipy import ndimage
from PIL import Image

STRIP = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
OUT   = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad"

mm = np.load(STRIP, mmap_mode='r')
H, W = mm.shape
lo, hi = np.percentile(np.asarray(mm[:, np.arange(2000, W-2000, 40)], np.float32), [1, 99])
to_v = lambda s: 1.0 - np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
g8 = lambda a: np.repeat((np.clip(a,0,1)*255).astype(np.uint8)[...,None],3,2)

def bg_mask(v):
    return ~ndimage.binary_dilation((v < 0.40) | (v > 0.97), iterations=4)

def illum_field(v, sz, sa):
    """content-masked normalized convolution: smooth over background only."""
    m = bg_mask(v).astype(np.float32)
    num = ndimage.gaussian_filter(v*m, (sz, sa))
    den = ndimage.gaussian_filter(m,   (sz, sa)) + 1e-6
    return num / den

def correct(v, sz, sa, alpha):
    F = illum_field(v, sz, sa)
    F = F / np.median(F)
    return np.clip(v / np.power(np.clip(F,1e-3,None), alpha), 0, 1), F

PAD, CW = 2500, 1150   # estimation pad, display width
SZ, ALPHA = 3, 1.0

def load_ctx(x0):
    a = max(0, x0-PAD); b = min(W, x0+CW+PAD)
    v = to_v(np.asarray(mm[:, a:b], np.float32))
    return v, x0-a                       # context + offset of display crop

# ladder over field-scale on the 'late' crop
v, off = load_ctx(170000)
cols = [g8(v[:, off:off+CW])]
for sa in [60, 100, 160]:
    c, _ = correct(v, SZ, sa, ALPHA)
    cols.append(g8(c[:, off:off+CW]))
    bb = np.nanstd(np.where(bg_mask(v[:,off:off+CW]), v[:,off:off+CW], np.nan))
    ba = np.nanstd(np.where(bg_mask(c[:,off:off+CW]), c[:,off:off+CW], np.nan))
    print(f"late sa={sa}: bg 2D std {bb:.4f} -> {ba:.4f}")
gap = np.full((H, 6, 3), 30, np.uint8)
row = cols[0]
for c in cols[1:]:
    row = np.concatenate([row, gap, c], 1)
Image.fromarray(row).save(f"{OUT}/geo2_ladder.png")   # original | sa60 | sa100 | sa160

# default correction on both crops (original | field | corrected) at sa=100
for name, x0 in [("mid_b", 120000), ("late", 170000)]:
    v, off = load_ctx(x0)
    c, F = correct(v, SZ, 60, ALPHA)
    Fv = (F - F.min())/(F.max()-F.min()+1e-6)
    Image.fromarray(np.concatenate([g8(v[:,off:off+CW]), gap, g8(Fv[:,off:off+CW]), gap, g8(c[:,off:off+CW])],1)).save(f"{OUT}/geo2_{name}.png")
print("saved")
