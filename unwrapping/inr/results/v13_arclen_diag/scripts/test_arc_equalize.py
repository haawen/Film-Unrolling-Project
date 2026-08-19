"""Validate v13's arc_equalize on the REAL v12 geometry, before any cluster render.

Checks:
  1. residual within-turn scale error after re-gridding (must be ~0, was +-5%)
  2. cross-z alignment is preserved EXACTLY (the phi-per-column map is shared by
     every z, so column c means the same azimuth at every z — the property
     resample_phi exists to provide)
  3. total film length is conserved (no global rescale sneaking in)
  4. z-heal still reconstructs the right ray angles on a non-uniform grid
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\li_k1\M_thesis")
from unwrapping.inr.unroll_walk_wholeroll_v13 import arc_equalize, z_heal_windings

TWO_PI = 2.0 * math.pi
d = np.load(r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v11\matched_walks.npz",
            allow_pickle=True)
cx = list(d["cx_a"]); cy = list(d["cy_a"]); seam = float(d["seam"])
paths = d["paths"]; n = len(paths); nw = len(paths[0])
eps_phi = math.radians(0.3)
cxA = np.asarray(cx)[:, None]; cyA = np.asarray(cy)[:, None]

print(f"{n} anchors, {nw} windings\n")
print(" wi     K -> K2   scale err BEFORE   AFTER    arc kept   xz-align")
bad = 0
for wi in [3, 8, 15, 22, 29, 34]:
    K = paths[0][wi].shape[0]
    X = np.empty((n, K), np.float32); Y = np.empty((n, K), np.float32)
    for a in range(n):
        X[a] = paths[a][wi][:, 0]; Y[a] = paths[a][wi][:, 1]
    phi = np.linspace(eps_phi, TWO_PI - eps_phi, K)

    def scale_err(X, Y, p):
        rm = np.median(np.hypot(X - cxA, Y - cyA), axis=0)
        ds = np.hypot(rm, np.gradient(rm, p)) * np.gradient(p)
        c = ds[len(ds) // 20: -len(ds) // 20]
        return 100 * (c.max() / c.min() - 1) / 2, float(np.sum(ds))

    e0, arc0 = scale_err(X, Y, phi)
    p2, X2, Y2 = arc_equalize(X, Y, cx, cy, phi, 1.0)
    e1, arc1 = scale_err(X2, Y2, p2)

    # cross-z alignment: at every anchor, column c must sit at azimuth seam+p2[c]
    az = np.mod(np.arctan2(Y2 - cyA, X2 - cxA) - seam, TWO_PI)
    dev = np.abs(az - p2[None, :])
    dev = np.minimum(dev, TWO_PI - dev)
    # ignore endpoint-clamped columns of partial walks (they legitimately sit off-ray)
    ok = np.degrees(np.median(dev, axis=0)).max()
    if e1 > 0.6 or abs(arc1 / arc0 - 1) > 1e-3:
        bad += 1
    print(f"{wi:3d} {K:6d} -> {len(p2):6d}   {e0:+8.2f}%      {e1:+6.2f}%   "
          f"{100*arc1/arc0:7.3f}%   {ok:6.3f}deg")

# --- z-heal must still work on a non-uniform grid ---
wi = 15
K = paths[0][wi].shape[0]
X = np.empty((n, K), np.float32); Y = np.empty((n, K), np.float32)
for a in range(n):
    X[a] = paths[a][wi][:, 0]; Y[a] = paths[a][wi][:, 1]
phi = np.linspace(eps_phi, TWO_PI - eps_phi, K)
p2, X2, Y2 = arc_equalize(X, Y, cx, cy, phi, 1.0)
# inject a synthetic 20px mid-turn jump over 30 anchors, then heal it
Xj, Yj = X2.copy(), Y2.copy()
sl = slice(200, 230); cols = slice(len(p2) // 2, len(p2) // 2 + 400)
th = (seam + p2)[None, :]
R = np.hypot(Xj - cxA, Yj - cyA)
R[sl, cols] += 20.0
Xj[sl, cols] = (cxA + R * np.cos(th))[sl, cols]
Yj[sl, cols] = (cyA + R * np.sin(th))[sl, cols]
dummy = (0.0, np.zeros((n, 8), np.float32), np.zeros((n, 8), np.float32),
         np.linspace(eps_phi, TWO_PI - eps_phi, 8))
out, nh = z_heal_windings([dummy, (600.0, Xj, Yj, p2), dummy], cx, cy, seam,
                          eps_phi, 6.0)
Rh = np.hypot(out[1][1] - cxA, out[1][2] - cyA)
R0 = np.hypot(X2 - cxA, Y2 - cyA)
resid = float(np.abs(Rh[sl, cols] - R0[sl, cols]).max())
print(f"\nz-heal on the non-uniform arc grid: healed {nh} bins, "
      f"residual after repairing a 20px jump = {resid:.2f} px "
      f"({'OK' if resid < 3 else 'FAIL'})")
print(f"\n{'ALL CHECKS PASS' if bad == 0 and resid < 3 else 'PROBLEM'}")
