"""Per-winding scale consistency: K columns are assigned from the RAW walked path
arc length, but the delivered curve's true arc (from the z-median r(phi)) differs.
If true_arc/K varies BETWEEN windings, every winding junction is a scale jump ->
framing steps once per turn."""
import math, sys
import numpy as np
TWO_PI = 2.0 * math.pi
d = np.load(sys.argv[1], allow_pickle=True)
cx = d["cx_a"]; cy = d["cy_a"]; paths = d["paths"]; n = len(paths); nw = len(paths[0])
print(" wi      K   true_arc   px/col   cum_err_px  cum_err_cells")
cum = 0.0; rat = []
for wi in range(nw):
    K = paths[0][wi].shape[0]
    R = np.empty((n, K), np.float32); ok = True
    for a in range(n):
        p = paths[a][wi]
        if p.shape[0] != K: ok = False; break
        R[a] = np.hypot(p[:, 0]-cx[a], p[:, 1]-cy[a])
    if not ok: continue
    rm = np.median(R, axis=0); dphi = TWO_PI/K
    arc = float(np.sum(np.hypot(rm, np.gradient(rm, dphi))*dphi))
    r = arc/K; cum += arc-K
    if 2 <= wi <= nw-3: rat.append(r)
    print(f"{wi:3d} {K:7d} {arc:10.0f} {r:8.4f} {cum:11.0f} {cum/903.6:13.2f}")
rat = np.array(rat)
print(f"\npx/col across windings: mean {rat.mean():.4f} std {rat.std():.4f} "
      f"min {rat.min():.4f} max {rat.max():.4f}")
print(f"  => winding-to-winding scale inconsistency = "
      f"{100*(rat.max()-rat.min()):.2f}% (a step at each junction)")
print(f"  => per-junction along-film position step: "
      f"{np.abs(np.diff(rat)).max()*np.median([paths[0][w].shape[0] for w in range(2,nw-2)]):.0f} px worst")
