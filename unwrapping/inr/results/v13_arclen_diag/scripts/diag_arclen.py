"""Does uniform-AZIMUTH column sampling stretch/compress the film cyclically?

For each winding: columns are a uniform grid in phi, so the film length each
column covers is ds/dphi = hypot(r, dr/dphi) — which varies with the winding's
eccentricity. Quantify the resulting local SCALE error vs a true arc-length
parameterization, and its period (should be ~once per turn = ~7 film cells).
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TWO_PI = 2.0 * math.pi
OUT = sys.argv[2] if len(sys.argv) > 2 else "arclen_diag.png"
d = np.load(sys.argv[1], allow_pickle=True)
cx = d["cx_a"]; cy = d["cy_a"]; paths = d["paths"]
n = len(paths); nw = len(paths[0])
PITCH = 903.6                       # px per film cell (v12 auto-refined)

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
tot_cols, tot_arc = 0, 0.0
scale_all = []
for wi in range(nw):
    K = paths[0][wi].shape[0]
    R = np.empty((n, K), np.float32)
    for a in range(n):
        p = paths[a][wi]
        if p.shape[0] != K:
            R = None
            break
        R[a] = np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])
    if R is None:
        continue
    rm = np.median(R, axis=0)                       # z-median profile r(phi)
    dphi = TWO_PI / K
    ds = np.hypot(rm, np.gradient(rm, dphi))        # px of film per radian
    arc = np.concatenate([[0.0], np.cumsum(ds[:-1] * dphi)])
    # what the render assumes: uniform px/col = arc[-1]/K
    scale = ds * dphi / (arc[-1] / K)               # local px-per-col / assumed
    tot_cols += K; tot_arc += arc[-1]
    if 2 <= wi <= nw - 3:
        scale_all.append(scale)
    if wi in (5, 15, 25, 33):
        axes[0].plot(np.linspace(0, 1, K), scale, lw=1,
                     label=f"w{wi} r~{np.median(rm):.0f}")
    # cumulative POSITION error in px (where a feature really is vs where it lands)
    err = arc - np.linspace(0, arc[-1], K)
    if wi in (5, 15, 25, 33):
        axes[1].plot(np.linspace(0, 1, K), err, lw=1, label=f"w{wi}")

axes[0].set_title("local film-px per column / assumed uniform  "
                  "(1.0 = correct; deviation = cyclic stretch of the picture)")
axes[0].set_xlabel("fraction of the turn"); axes[0].axhline(1, color="k", lw=0.5)
axes[0].legend(fontsize=8)
axes[1].set_title("cumulative along-film POSITION error (px) from uniform-phi "
                  f"columns  [1 film cell = {PITCH:.0f}px]")
axes[1].set_xlabel("fraction of the turn"); axes[1].axhline(0, color="k", lw=0.5)
axes[1].legend(fontsize=8)

S = np.array([np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, len(s)), s)
              for s in scale_all])
axes[2].plot(np.linspace(0, 1, 400), S.mean(0), "b", lw=2, label="mean over windings")
axes[2].fill_between(np.linspace(0, 1, 400), S.min(0), S.max(0), alpha=0.2)
axes[2].axhline(1, color="k", lw=0.5)
axes[2].set_title("scale modulation is COHERENT across windings "
                  "(= eccentricity, once per turn)")
axes[2].set_xlabel("fraction of the turn"); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT, dpi=110)

print(f"total cols {tot_cols}, total true arc {tot_arc:.0f} px "
      f"({100*(tot_arc-tot_cols)/tot_cols:+.2f}% vs col count)")
print(f"local scale: mean {S.mean():.4f}  min {S.min():.4f}  max {S.max():.4f}")
print(f"  => picture is stretched/compressed by "
      f"{100*(S.max()-S.min())/2:.1f}% peak-to-mean within each turn")
print(f"cells per turn ~ {tot_arc/nw/PITCH:.1f}  => the modulation period is "
      f"~{tot_arc/nw/PITCH:.1f} film frames")
mx = max(np.abs(np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, len(s)),
                          np.cumsum(s - 1) * (tot_arc / nw / len(s)))).max()
         for s in scale_all)
print(f"peak along-film position error within a turn ~ {mx:.0f} px "
      f"= {mx/PITCH:.2f} film cells")
print(f"saved {OUT}")
