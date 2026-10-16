"""Measure the SEAM GAP in a walk's delivered geometry (matched_walks.npz).

Why: the fit overlay for the new scan shows an untraced wedge far wider than the
structural 2*--seam-exclude-deg (12 deg). Because the stored `paths` are already
on the common phi grid (resample_phi), a partial walk is ENDPOINT-CLAMPED: the
repeated endpoint draws as nothing, so a wide gap == clamped bins == walks that
never reached the seam target.

Two outputs:
  (1) per-winding real (non-clamped) azimuthal span, so "how big is the gap"
      becomes a number instead of an eyeball;
  (2) the ABSOLUTE azimuth at which the walks stop. If the seam target is wrong,
      walks stop where the film is physically discontinuous, not where they were
      told -- so a tight cluster of stop-azimuths away from the target IS an
      independent seam estimate (and a tight cluster AT the target means the
      target is fine and something else truncates the walks).

Run on any matched_walks.npz (new scan or an old-scan control).
"""
import argparse
import math

import numpy as np

TWO_PI = 2 * math.pi


def clamp_mask(P, tol=1e-6):
    """True where a bin is an endpoint-clamp duplicate (leading/trailing runs)."""
    n = len(P)
    same = np.zeros(n, bool)
    d = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    dup = d < tol
    lead = 0
    while lead < n - 1 and dup[lead]:
        lead += 1
    trail = 0
    while trail < n - 1 and dup[n - 2 - trail]:
        trail += 1
    same[:lead] = True
    if trail:
        same[n - trail:] = True
    return same


def crossings(paths, cx_a, cy_a, seam, eps_deg, idx, min_sep):
    """Where do adjacent windings collide? = where a walk jumped onto a neighbour.

    THE SEAM PROBE. With a small --seam-exclude-deg the walk is forced THROUGH the
    physical seam, so a wrong seam angle shows up as adjacent windings converging /
    swapping order at the REAL seam azimuth. Radial spacing is ~26px (new scan) /
    ~21px (old), so a bin under `min_sep` is a genuine collision, and the azimuth
    histogram of those bins localises the seam.
    """
    W = len(paths[idx[0]])
    n_bad = np.zeros(0)
    hist = None
    nb = 120
    hist = np.zeros(nb)
    tot = np.zeros(nb)
    worst = []
    for a in idx:
        R = []
        for w in range(W):
            P = np.asarray(paths[a][w], float)
            R.append(np.hypot(P[:, 0] - cx_a[a], P[:, 1] - cy_a[a]))
        L = min(len(r) for r in R)
        R = np.array([r[:L] for r in R])
        phi = np.linspace(eps_deg, 360.0 - eps_deg, L)
        b = np.clip((phi / 360.0 * nb).astype(int), 0, nb - 1)
        gap = np.diff(R, axis=0)                      # (W-1, L), >0 if ordered
        bad = gap < min_sep
        for k in range(nb):
            m = b == k
            tot[k] += m.sum() * (W - 1)
            hist[k] += bad[:, m].sum()
        for w in range(W - 1):
            if bad[w].any():
                worst.append((w, float(gap[w].min()), float(phi[np.argmin(gap[w])])))
    frac = np.where(tot > 0, hist / np.maximum(tot, 1), 0.0)
    print(f"\ncollision rate per 3-deg azimuth bin (adjacent-winding gap < {min_sep}px):")
    top = np.argsort(frac)[::-1][:10]
    for k in sorted(top):
        print(f"  az {k * 360 / nb:5.1f}-{(k + 1) * 360 / nb:5.1f} deg   "
              f"{100 * frac[k]:6.2f}%   {'#' * int(round(60 * frac[k] / max(frac.max(), 1e-9)))}")
    print(f"  overall collision rate {100 * hist.sum() / max(tot.sum(), 1):.3f}%; "
          f"peak bin az {np.argmax(frac) * 360 / nb:.1f} deg at {100 * frac.max():.2f}%")
    if worst:
        worst.sort(key=lambda t: t[1])
        print("  worst 5 (winding-pair, min gap px, azimuth deg):")
        for w, g, p in worst[:5]:
            print(f"    w{w:02d}/w{w + 1:02d}  {g:7.1f}px  at {p:6.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--eps-deg", type=float, default=6.0,
                    help="the --seam-exclude-deg the walk ran with")
    ap.add_argument("--anchors", type=int, default=8, help="how many anchors to sample")
    ap.add_argument("--min-sep", type=float, default=8.0,
                    help="adjacent-winding radial gap below this = a collision "
                         "(spacing is ~26px new scan / ~21px old)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    paths = d["paths"]
    cx_a = d["cx_a"]
    cy_a = d["cy_a"]
    seam = float(d["seam"])
    n = len(paths)
    W = len(paths[0])
    print(f"{n} anchors x {W} windings; seam = {math.degrees(seam):.2f} deg; "
          f"eps = {args.eps_deg} deg  (structural gap {2 * args.eps_deg:.0f} deg)")

    idx = np.unique(np.linspace(0, n - 1, min(args.anchors, n)).astype(int))
    spans = np.full((len(idx), W), np.nan)
    lo_stop, hi_stop = [], []          # absolute azimuth (deg) where walks stop
    for ai, a in enumerate(idx):
        for w in range(W):
            P = np.asarray(paths[a][w], float)
            if len(P) < 8 or not np.isfinite(P).all():
                continue
            cl = clamp_mask(P)
            good = ~cl
            if good.sum() < 8:
                continue
            # phi grid is monotone over [eps, 2pi-eps] by construction
            phi = np.linspace(args.eps_deg, 360.0 - args.eps_deg, len(P))
            spans[ai, w] = phi[good][-1] - phi[good][0]
            th = np.degrees(np.arctan2(P[:, 1] - cy_a[a], P[:, 0] - cx_a[a]))
            if cl[0]:
                lo_stop.append(th[good][0] % 360)
            if cl[-1]:
                hi_stop.append(th[good][-1] % 360)

    full = 360.0 - 2 * args.eps_deg
    print("\nper-winding real azimuthal span (deg), median over sampled anchors:")
    med = np.nanmedian(spans, axis=0)
    for w in range(W):
        bar = "#" * int(round(30 * (med[w] / full))) if np.isfinite(med[w]) else ""
        print(f"  w{w:02d}  {med[w]:6.1f} / {full:.0f}   {bar}")
    ok = np.isfinite(med)
    print(f"\nspan: median {np.nanmedian(med):.1f}  min {np.nanmin(med):.1f}  "
          f"max {np.nanmax(med):.1f} of {full:.0f} deg "
          f"({(med[ok] > full - 1).sum()}/{ok.sum()} windings reach the target both ways)")
    print(f"total gap = {full - np.nanmedian(med):.1f} deg beyond the structural "
          f"{2 * args.eps_deg:.0f} deg, on the median winding")

    def circ(v, name):
        if not v:
            print(f"  {name}: none (no clamped ends)")
            return
        a = np.radians(v)
        m = math.degrees(np.angle(np.mean(np.exp(1j * a)))) % 360
        R = abs(np.mean(np.exp(1j * a)))
        dev = np.degrees(np.abs((np.radians(v) - math.radians(m) + math.pi)
                                % TWO_PI - math.pi))
        print(f"  {name}: n={len(v):3d}  circ-mean {m:6.1f} deg  R={R:.3f}  "
              f"MAD {np.median(dev):5.1f} deg")

    print(f"\nABSOLUTE azimuth where walks stop (target seam "
          f"{math.degrees(seam) % 360:.1f} deg):")
    circ(lo_stop, "CW  end")
    circ(hi_stop, "CCW end")
    circ(lo_stop + hi_stop, "both  ")
    print("  CAVEAT: this estimator FAILED its control (old scan, true seam 60deg, "
          "gives the same 251/349deg clusters = partial film-end windings). "
          "Do not read a seam off it.")

    crossings(paths, cx_a, cy_a, seam, args.eps_deg, idx, args.min_sep)


if __name__ == "__main__":
    main()
