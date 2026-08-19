"""Characterize the BROAD brightness variation (banding / shading) on the strip,
to decide destriping vs shading-correction vs deflicker.

strip = (1184, 204521) = (z = film WIDTH rows, arc = film LENGTH cols).
'per-slice intensity variation' = per-ROW gain -> horizontal stripes.
"""
import numpy as np
from scipy import ndimage

p = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
a = np.load(p, mmap_mode='r')
H, W = a.shape
print("strip", a.shape)

win = np.asarray(a[:, 90000:96000], np.float32)   # 1184 x 6000 mid-roll block
lo, hi = np.percentile(win, [1, 99])
n = np.clip((win - lo) / (hi - lo + 1e-6), 0, 1)

# robust per-ROW (per z-slice) and per-COL levels via median (ignores Mickey)
row_med = np.median(n, axis=1)     # length H=1184  -> striping along z
col_med = np.median(n, axis=0)     # length 6000    -> shading along arc

def band_energy(x):
    x = x - x.mean()
    smooth = ndimage.gaussian_filter1d(x, 8)      # broad trend
    fast = x - smooth                             # row-to-row jitter (stripes)
    return x.std(), smooth.std(), fast.std()

print("\n-- per-ROW median (z / film-width direction) --")
t, s, f = band_energy(row_med)
print(f"  total std {t:.4f} | broad-shading std {s:.4f} | fast row-jitter(stripe) std {f:.4f}")
print(f"  row-to-row |diff| median {np.median(np.abs(np.diff(row_med))):.4f}, max {np.abs(np.diff(row_med)).max():.4f}")

print("\n-- per-COL median (arc / film-length direction) --")
t, s, f = band_energy(col_med)
print(f"  total std {t:.4f} | broad std {s:.4f} | fast std {f:.4f}")

# 2D low-frequency shading field vs local content contrast
lowf = ndimage.gaussian_filter(n, 40)             # broad brightness field
content = n - lowf
print("\n-- 2D decomposition --")
print(f"  broad shading field (sigma40): range {lowf.min():.3f}..{lowf.max():.3f}, std {lowf.std():.4f}")
print(f"  residual (content+spots+stripe): std {content.std():.4f}")
# directionality of the mid-freq residual: horizontal vs vertical gradient energy
gy = np.abs(np.diff(n, axis=0)).mean()   # vertical gradient (across z rows)
gx = np.abs(np.diff(n, axis=1)).mean()   # horizontal gradient (across arc)
print(f"  mean |grad| across-rows(z) {gy:.4f}  vs across-cols(arc) {gx:.4f}  (stripe if row-grad >> col-grad in flat areas)")

# same but in a FLAT background sub-block (rows 200-500 tend to be background)
flat = n[150:520, :]
gy2 = np.abs(np.diff(flat, axis=0)).mean()
gx2 = np.abs(np.diff(flat, axis=1)).mean()
print(f"  FLAT block: across-rows(z) {gy2:.4f} vs across-cols(arc) {gx2:.4f}")
