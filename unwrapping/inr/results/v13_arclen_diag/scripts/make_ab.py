"""Full-resolution A/B for inspection: the same run of film cells cut at a fixed
pitch from the v12 strip, and from that strip after the v13 arc-length re-grid.

A red guide line sits at the SAME position inside every cell. Under v12 the
picture slides against that line and back within one turn (the framing creep);
under v13 it should stay put.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

SP = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\v13_arclen_diag"
os.makedirs(OUT, exist_ok=True)
TWO_PI = 2.0 * math.pi
EPS = math.radians(0.3)
WI = 8

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
blk = np.asarray(S[:, starts[WI]: starts[WI] + K], np.float32)      # full z
phi = np.linspace(EPS, TWO_PI - EPS, K)
R = np.hypot(np.stack([paths[a][WI][:, 0] for a in range(n)]) - cx,
             np.stack([paths[a][WI][:, 1] for a in range(n)]) - cy)
rm = np.median(R, axis=0)
ds = np.hypot(rm, np.gradient(rm, phi))
s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phi))])
K2 = max(2, int(round(s[-1])))
src = np.interp(np.interp(np.linspace(0, s[-1], K2), s, phi), phi, np.arange(K))
blk13 = np.empty((blk.shape[0], K2), np.float32)
for r in range(blk.shape[0]):
    blk13[r] = np.interp(src, np.arange(K), blk[r])
print(f"winding {WI}: v12 {blk.shape} -> v13 {blk13.shape}")


def refine_pitch(B):
    p = B.mean(0); p = p - gaussian_filter1d(p, 1500)
    lags = np.arange(820, 990)
    ac = np.array([np.dot(p[:-L], p[L:]) / (len(p) - L) for L in lags])
    k = int(np.argmax(ac)); y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    dd = y0 - 2 * y1 + y2
    return lags[k] + (0.5 * (y0 - y2) / dd if abs(dd) > 1e-12 else 0.0)


def strip_of_cells(B, pitch, ncell=5, start=0):
    lo, hi = np.percentile(B, [1, 99])
    tiles = []
    for i in range(start, start + ncell):
        c0 = int(round(20 + i * pitch)); W = int(pitch)
        t = B[:, c0:c0 + W]
        if t.shape[1] < W:
            break
        t = np.clip((t - lo) / (hi - lo + 1e-6), 0, 1)
        tiles.append((255 * (1 - t)).astype(np.uint8))          # negative -> positive
    return tiles


p12, p13 = refine_pitch(blk), refine_pitch(blk13)
t12, t13 = strip_of_cells(blk, p12), strip_of_cells(blk13, p13)
nc = min(len(t12), len(t13))
H = t12[0].shape[0]
GAP, LAB = 10, 46
Wc = min(min(t.shape[1] for t in t12), min(t.shape[1] for t in t13))
canvas = Image.new("RGB", (nc * (Wc + GAP), 2 * H + 3 * GAP + 2 * LAB), "white")
dr = ImageDraw.Draw(canvas)
for row, (tiles, tag, pit) in enumerate(
        [(t12, f"v12  uniform-azimuth columns  (pitch {p12:.1f})", p12),
         (t13, f"v13  arc-length columns  (pitch {p13:.1f})", p13)]):
    y = LAB + row * (H + GAP + LAB)
    dr.text((4, y - 32), tag, fill="black")
    for i in range(nc):
        x = i * (Wc + GAP)
        canvas.paste(Image.fromarray(tiles[i][:, :Wc]), (x, y))
        dr.line([(x + Wc // 2, y), (x + Wc // 2, y + H)], fill=(255, 0, 0), width=3)
canvas.save(os.path.join(OUT, "AB_cells_fullres.png"))
half = canvas.resize((canvas.width // 2, canvas.height // 2))
half.save(os.path.join(OUT, "AB_cells_half.png"))
print(f"saved AB_cells_fullres.png {canvas.size}")

# also the raw single-version montages at full res, for close inspection
for tiles, name in [(t12, "cells_v12_fullres.png"), (t13, "cells_v13_fullres.png")]:
    im = np.concatenate([np.pad(t[:, :Wc], ((0, 0), (0, 8)), constant_values=255)
                         for t in tiles], axis=1)
    Image.fromarray(im).save(os.path.join(OUT, name))
print("saved full-res per-version montages")
