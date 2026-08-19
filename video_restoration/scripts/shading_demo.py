"""Brightness-variation decomposition on strip crops (video-positive):
  original | estimated broad shading field | flat-fielded (quick classical)
Standalone (numpy/scipy/PIL). Purpose: show 'big areas of brightness change'.
"""
import numpy as np
from scipy import ndimage
from PIL import Image

STRIP = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
OUT   = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad"

mm = np.load(STRIP, mmap_mode='r')
cols = np.linspace(2000, mm.shape[1]-2000, 40).astype(int)
lo, hi = np.percentile(np.asarray(mm[:, cols], np.float32), [1, 99])

def to_video(strip):
    return 1.0 - np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)  # --invert

g8 = lambda a: np.repeat((np.clip(a,0,1)*255).astype(np.uint8)[...,None],3,2)

def field_estimate(v):
    # broad illumination field: heavy median (ignores spots) then Gaussian (slow shading).
    # NOTE: also contains genuine content low-freq -> flat-fielding flattens real tone too.
    return ndimage.gaussian_filter(ndimage.median_filter(v, size=41), 60)

for name, x0 in [("mid_b", 120000), ("late", 170000)]:
    v = to_video(np.asarray(mm[:, x0:x0+1150], np.float32))
    S = field_estimate(v)
    corr = np.clip(v / (S + 1e-3) * S.mean(), 0, 1)      # homomorphic-style divide
    Sv = (S - S.min()) / (S.max() - S.min() + 1e-6)      # stretch field for display
    gap = np.full((v.shape[0], 6, 3), 30, np.uint8)
    Image.fromarray(np.concatenate([g8(v), gap, g8(Sv), gap, g8(corr)], 1)).save(f"{OUT}/shading_{name}.png")
    print(f"{name}: field range {S.min():.3f}..{S.max():.3f} std {S.std():.3f}")
print("saved")
