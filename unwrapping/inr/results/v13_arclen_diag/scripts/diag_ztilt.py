"""The OTHER axis: rows of the strip are CT z-slices, i.e. the render assumes the
film's width axis is parallel to z. It isn't — the emulsion surface at a fixed
(winding, azimuth) moves radially as z advances (dr/dz != 0), so the film is
TILTED w.r.t. z and one z-step covers sqrt(1+(dr/dz)^2) of film width.

If that factor varies with azimuth, the picture's HEIGHT breathes along the film
= the measured per-frame scale-y wobble (+-2.3%). Quantify it.
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TWO_PI = 2.0 * math.pi
d = np.load(sys.argv[1], allow_pickle=True)
z = np.asarray(d["z_anchor"], float)
cx = d["cx_a"]; cy = d["cy_a"]; paths = d["paths"]
n = len(paths); nw = len(paths[0])
OUT = sys.argv[2]

print(f"anchor z spacing: med {np.median(np.diff(z)):.2f}, "
      f"max {np.diff(z).max():.0f} (chunk gaps)")

fig, axes = plt.subplots(2, 1, figsize=(13, 7))
facs, sways = [], []
for wi in range(2, nw - 2):
    K = paths[0][wi].shape[0]
    R = np.empty((n, K), np.float32)
    ok = True
    for a in range(n):
        p = paths[a][wi]
        if p.shape[0] != K:
            ok = False; break
        R[a] = np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])
    if not ok:
        continue
    # dr/dz per (anchor, phi) — use only WITHIN-chunk neighbours (dz small)
    dz = np.diff(z)[:, None]
    good = (dz[:, 0] > 0) & (dz[:, 0] < 4)          # skip chunk gaps
    drdz = np.diff(R, axis=0)[good] / dz[good]
    # robust per-azimuth slope
    s_phi = np.median(drdz, axis=0)                 # (K,) local tilt vs azimuth
    fac = np.sqrt(1.0 + s_phi ** 2)                 # film-width px per z-step
    facs.append(np.interp(np.linspace(0, 1, 400), np.linspace(0, 1, K), fac))
    # SWAY: how far the emulsion moves radially over the full z range
    sways.append(float(np.abs(np.median(R[-20:], axis=0) -
                              np.median(R[:20], axis=0)).max()))
    if wi in (5, 15, 25, 33):
        axes[0].plot(np.linspace(0, 1, K), s_phi, lw=0.8, label=f"w{wi}")

F = np.array(facs)
axes[0].set_title("local dr/dz (px of radius per CT slice) vs azimuth "
                  "— nonzero => film tilted w.r.t. z")
axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(fontsize=8)
axes[0].set_xlabel("fraction of the turn")
axes[1].plot(np.linspace(0, 1, 400), F.mean(0), "r", lw=2)
axes[1].fill_between(np.linspace(0, 1, 400), np.percentile(F, 10, axis=0),
                     np.percentile(F, 90, axis=0), alpha=0.2, color="r")
axes[1].axhline(1, color="k", lw=0.5)
axes[1].set_title("film-width px per z-step = sqrt(1+(dr/dz)^2)  "
                  "(1.0 = rows are true film width)")
axes[1].set_xlabel("fraction of the turn")
plt.tight_layout(); plt.savefig(OUT, dpi=110)

print(f"\ndr/dz  : |med| {np.median(np.abs(F - 1)) and 0 or 0:.0f}"
      f"  p50 {np.median(np.sqrt(F**2-1)):.3f}  p95 "
      f"{np.percentile(np.sqrt(np.clip(F**2-1,0,None)), 95):.3f} px/slice")
print(f"width factor sqrt(1+(dr/dz)^2): mean {F.mean():.4f}  "
      f"p50 {np.median(F):.4f}  p95 {np.percentile(F,95):.4f}  max {F.max():.4f}")
print(f"  => row scale (picture height) varies by "
      f"{100*(np.percentile(F,95)-np.percentile(F,5))/2:.2f}% (p5..p95) "
      f"across azimuth")
print(f"radial SWAY over the full z range: med {np.median(sways):.0f} px, "
      f"max {max(sways):.0f} px")
print(f"saved {OUT}")
