"""Apply geometry-aware radiometric normalization to the FULL v9 strip.
Anisotropic content-masked illumination field (sz=3 along z, sa=60 along arc),
divided out toward a single GLOBAL background level so chunks stay consistent.
Processed in overlapping column chunks (memory-safe). Saves a cleaned strip in
the SAME value domain as wholeroll.npy so make_film_video consumes it unchanged.
"""
import numpy as np, time
from scipy import ndimage

SRC = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
DST = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll_geonorm.npy"
SZ, SA, ALPHA = 3.0, 60.0, 1.0
CHUNK, PAD = 20000, 300

mm = np.load(SRC, mmap_mode='r')
H, W = mm.shape
print("strip", mm.shape)

# global scale + global background target level (from a column subsample)
cols_s = np.arange(2000, W-2000, 30)
samp = np.asarray(mm[:, cols_s], np.float32)
lo, hi = np.percentile(samp, [1, 99])
to_v = lambda s: 1.0 - np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
vs = to_v(samp)
bg = ~ndimage.binary_dilation((vs < 0.40) | (vs > 0.97), iterations=4)
G = float(np.median(vs[bg]))
print(f"lo/hi {lo:.4f}/{hi:.4f}  global bg level G={G:.4f}")

out = np.lib.format.open_memmap(DST, mode='w+', dtype=np.float32, shape=(H, W))
t0 = time.time()
for a in range(0, W, CHUNK):
    b = min(W, a + CHUNK)
    a2, b2 = max(0, a-PAD), min(W, b+PAD)
    s = np.asarray(mm[:, a2:b2], np.float32)
    v = to_v(s)
    m = (~ndimage.binary_dilation((v < 0.40) | (v > 0.97), iterations=4)).astype(np.float32)
    num = ndimage.gaussian_filter(v*m, (SZ, SA))
    den = ndimage.gaussian_filter(m,   (SZ, SA))
    F = np.where(den > 1e-3, num/np.maximum(den,1e-6), G)      # local illumination; fallback=G (no corr)
    F = np.maximum(F, 1e-3)
    v_corr = np.clip(v * np.power(G/F, ALPHA), 0, 1)            # multiplicative de-shade toward G
    s_corr = lo + (hi - lo) * (1.0 - v_corr)                    # back to strip domain
    out[:, a:b] = s_corr[:, (a-a2):(a-a2)+(b-a)]
    print(f"  cols {a}-{b}  ({time.time()-t0:.0f}s)", flush=True)
out.flush()
print("saved", DST)
