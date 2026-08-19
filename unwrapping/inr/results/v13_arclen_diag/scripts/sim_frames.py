"""Picture-level A/B, still zero cluster time.

Cut film cells at a FIXED pitch (what make_film_video --phase-lock global does)
from one winding of the v12 strip, and from the same winding after the v13
arc-length re-grid. Then measure how far each cell's content sits from where the
fixed-pitch grid puts it. Under uniform-azimuth columns that offset should sweep
back and forth across the turn (= the framing creep you see as wobble); under
arc-length columns it should stay put.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

SP = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 2.0 * math.pi
EPS = math.radians(0.3)
WI = 8                                   # a mid-inner winding (strong modulation)

S = np.load(r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v12\wholeroll.npy",
            mmap_mode="r")
d = np.load(r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v11\matched_walks.npz",
            allow_pickle=True)
cx = np.asarray(d["cx_a"], float)[:, None]; cy = np.asarray(d["cy_a"], float)[:, None]
paths = d["paths"]; n = len(paths); nw = len(paths[0])
Ks = [paths[0][w].shape[0] for w in range(nw)]
Ms = [int(round(2 * EPS * Ks[i] / (TWO_PI - 2 * EPS))) for i in range(nw - 1)] + [0]
starts = np.cumsum([0] + [Ks[i] + Ms[i] for i in range(nw)])

K = Ks[WI]
blk = np.asarray(S[::2, starts[WI]: starts[WI] + K], np.float32)   # (Z/2, K)
print(f"winding {WI}: block {blk.shape}")

phi = np.linspace(EPS, TWO_PI - EPS, K)
R = np.hypot(np.stack([paths[a][WI][:, 0] for a in range(n)]) - cx,
             np.stack([paths[a][WI][:, 1] for a in range(n)]) - cy)
rm = np.median(R, axis=0)
ds = np.hypot(rm, np.gradient(rm, phi))
s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phi))])
K2 = max(2, int(round(s[-1])))
phi_new = np.interp(np.linspace(0, s[-1], K2), s, phi)
src = np.interp(phi_new, phi, np.arange(K))           # v12 column -> v13 column
blk13 = np.empty((blk.shape[0], K2), np.float32)
for r in range(blk.shape[0]):
    blk13[r] = np.interp(src, np.arange(K), blk[r])
print(f"re-gridded to {blk13.shape}")


def cell_offsets(B, tag):
    """Cut cells at the best global uniform pitch; report how far each cell's
    content actually sits from its slot (sub-px, by cross-correlation)."""
    p = B.mean(0); p = p - gaussian_filter1d(p, 1500)
    lags = np.arange(820, 990)
    ac = np.array([np.dot(p[:-L], p[L:]) / (len(p) - L) for L in lags])
    k = int(np.argmax(ac))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]; dd = y0 - 2 * y1 + y2
    pitch = lags[k] + (0.5 * (y0 - y2) / dd if abs(dd) > 1e-12 else 0.0)
    ncell = int((B.shape[1] - 40) // pitch)
    ref = None; offs = []
    for i in range(ncell):
        c0 = int(round(20 + i * pitch))
        cell = p[c0:c0 + int(pitch)]
        if len(cell) < int(pitch):
            break
        if ref is None:
            ref = cell; offs.append(0.0); continue
        # sub-pixel shift of this cell's frame-line pattern vs the first cell
        m = min(len(ref), len(cell)); w = 120
        cc = np.array([np.dot(ref[w:m - w], cell[w + t:m - w + t])
                       for t in range(-w, w + 1)])
        j = int(np.argmax(cc))
        if 0 < j < len(cc) - 1:
            a0, a1, a2 = cc[j - 1], cc[j], cc[j + 1]; dd2 = a0 - 2 * a1 + a2
            f = 0.5 * (a0 - a2) / dd2 if abs(dd2) > 1e-12 else 0.0
        else:
            f = 0.0
        offs.append((j - w) + f)
    offs = np.array(offs)
    print(f"  {tag:22s} pitch {pitch:7.2f}  cells {len(offs):3d}  "
          f"framing offset: ptp {np.ptp(offs):6.1f}px  std {offs.std():5.1f}px")
    return pitch, offs


print("\nframing creep within ONE turn (fixed-pitch cutting):")
p12, o12 = cell_offsets(blk, "v12 uniform-azimuth")
p13, o13 = cell_offsets(blk13, "v13 arc-length")
print(f"\n  => creep reduced {np.ptp(o12)/max(np.ptp(o13),1e-6):.1f}x "
      f"({np.ptp(o12):.0f}px -> {np.ptp(o13):.0f}px of a {p13:.0f}px cell "
      f"= {100*np.ptp(o12)/p12:.1f}% -> {100*np.ptp(o13)/p13:.1f}% of frame width)")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(o12, "o-", color="C3", label=f"v12 uniform-azimuth (ptp {np.ptp(o12):.0f}px)")
ax.plot(o13, "o-", color="C0", label=f"v13 arc-length (ptp {np.ptp(o13):.0f}px)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("film cell within one turn"); ax.set_ylabel("framing offset (px)")
ax.set_title(f"Where each cell's content lands vs its fixed-pitch slot (winding {WI})")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(SP, "sim_frames_creep.png"), dpi=110)

# montage: same 6 consecutive cells, v12 above / v13 below
from PIL import Image
def montage(B, pitch, path, ncell=6, start=1):
    H = B.shape[0]; W = int(pitch)
    lo, hi = np.percentile(B, [1, 99])
    tiles = []
    for i in range(start, start + ncell):
        c0 = int(round(20 + i * pitch))
        t = np.clip((B[:, c0:c0 + W] - lo) / (hi - lo + 1e-6), 0, 1)
        if t.shape[1] < W:
            break
        tiles.append((255 * (1 - t)).astype(np.uint8))       # invert (negative)
    im = np.concatenate([np.pad(t, ((0, 0), (0, 6)), constant_values=255)
                         for t in tiles], axis=1)
    Image.fromarray(im).resize((im.shape[1] // 3, im.shape[0] // 3)).save(path)
    return im.shape

print("montage", montage(blk, p12, os.path.join(SP, "cells_v12.png")))
print("montage", montage(blk13, p13, os.path.join(SP, "cells_v13.png")))
print("saved sim_frames_creep.png, cells_v12.png, cells_v13.png")
