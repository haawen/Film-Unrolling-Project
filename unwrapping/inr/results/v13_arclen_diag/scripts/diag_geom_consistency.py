"""Quantify the three candidate sources of residual video wobble, on a cached
matched_walks.npz (the DELIVERABLE per-anchor winding curves, resampled on the
common phi grid).

Sources tested
  A. ARC-LENGTH BREATHING: columns are uniform in AZIMUTH, so px-of-film per
     column = ds/dphi = hypot(r, dr/dphi) varies around each turn (eccentricity).
     -> the strip's horizontal scale breathes -> per-frame width/scale-x wobble.
  B. PER-ANCHOR HF JITTER: r(phi, z) minus a smooth surface = independent
     per-slice walk noise that passes straight into the strip.
  C. CENTER DRIFT across z (context for both).
"""
import math
import sys

import numpy as np

TWO_PI = 2.0 * math.pi
PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "unwrapping/inr/results/walk_dense_v9/matched_walks.npz"

d = np.load(PATH, allow_pickle=True)
z = d["z_anchor"]; cx = d["cx_a"]; cy = d["cy_a"]; seam = float(d["seam"])
paths = d["paths"]                      # (n,) object -> list of (K,2) per winding
n = len(paths); nw = len(paths[0])
print(f"{n} anchors, {nw} windings, z {z.min()}..{z.max()}, "
      f"seam {math.degrees(seam):.1f}deg")
print(f"center drift: cx {cx.min():.1f}..{cx.max():.1f} (ptp {np.ptp(cx):.1f}), "
      f"cy ptp {np.ptp(cy):.1f}")

# per-winding: build R(n, K) about the per-anchor center
print("\nwi     K   r_med   ecc(ptp)  ds/dphi ratio  |  HF jitter px (r - smooth)")
print("                                max/min      |   rms    p95    max")
rows = []
for wi in range(nw):
    K = paths[0][wi].shape[0]
    R = np.empty((n, K), np.float32)
    ok = True
    for a in range(n):
        p = paths[a][wi]
        if p.shape[0] != K:
            ok = False
            break
        R[a] = np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])
    if not ok:
        print(f"{wi:2d}  ragged K, skipped")
        continue

    # ---- A. arc-length breathing on the z-median profile ----
    rm = np.median(R, axis=0)
    dphi = TWO_PI / K
    drdphi = np.gradient(rm, dphi)
    ds = np.hypot(rm, drdphi)                    # px of film per radian
    # ignore the extreme ends (partial-walk clamps)
    core = ds[K // 20: -K // 20] if K > 40 else ds
    ratio = core.max() / max(core.min(), 1e-6)

    # ---- B. HF jitter: r minus a smooth (phi,z) surface ----
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(R, size=(31, 51), mode="nearest")
    res = (R - smooth)[:, K // 20: -K // 20] if K > 40 else (R - smooth)
    rows.append((wi, K, float(np.median(rm)), float(np.ptp(rm)), ratio,
                 float(np.sqrt((res ** 2).mean())),
                 float(np.percentile(np.abs(res), 95)), float(np.abs(res).max())))
    wi_, K_, rmed, ecc, rat, rms, p95, mx = rows[-1]
    print(f"{wi:2d} {K:6d} {rmed:7.0f} {ecc:9.1f} {rat:13.3f}  | "
          f"{rms:6.2f} {p95:6.2f} {mx:6.1f}")

arr = np.array([r[4] for r in rows])
jit = np.array([r[5] for r in rows])
print(f"\nSUMMARY over {len(rows)} windings")
print(f"  ds/dphi max/min ratio : med {np.median(arr):.3f}  "
      f"p90 {np.percentile(arr, 90):.3f}  max {arr.max():.3f}")
print(f"  => horizontal scale breathes +-{100*(np.median(arr)-1)/2:.1f}% "
      f"within a turn (uniform-phi columns)")
print(f"  HF radial jitter rms  : med {np.median(jit):.2f} px  max {jit.max():.2f} px")
