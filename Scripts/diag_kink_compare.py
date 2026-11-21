"""Local KINKS in two matched_walks.npz: per (anchor, winding), the largest
deviation of r(column) from its own along-turn running median (win columns).
A V-notch / step of a few columns is invisible to whole-row medians but is what
renders as a local tear. Reports counts above thresholds and where the second
file has kinks the first does not (and vice versa)."""
import argparse
import numpy as np
from scipy.ndimage import median_filter


def kinks(path, win):
    d = np.load(path, allow_pickle=True)
    z = np.asarray(d["z_anchor"], float)
    cx = np.asarray(d["cx_a"], float); cy = np.asarray(d["cy_a"], float)
    P = d["paths"]; n, nw = len(z), len(P[0])
    K = np.zeros((n, nw), np.float32)
    for k in range(1, nw - 1):                  # film ends are partial: skip
        for a in range(n):
            p = np.asarray(P[a][k], float)
            r = np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])
            K[a, k] = np.abs(r - median_filter(r, win, mode="nearest")).max()
    return z, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--win", type=int, default=61)
    args = ap.parse_args()
    z, KA = kinks(args.old, args.win)
    _, KB = kinks(args.new, args.win)
    for t in (5, 8, 13):
        print(f"kinks > {t:2d}px: old {int((KA > t).sum()):6d}   new {int((KB > t).sum()):6d}")
    for lab, M in [("NEW-only (new>8, old<4)", (KB > 8) & (KA < 4)),
                   ("REMOVED (old>8, new<4)", (KA > 8) & (KB < 4))]:
        a, k = np.nonzero(M)
        print(f"{lab}: {len(a)}")
        if len(a):
            bands = np.histogram(z[a], bins=[470, 500, 600, 700, 760, 800, 1000, 1400, 1950])
            print("   per z band:", dict(zip([f"{lo}-{hi}" for lo, hi in
                  zip(bands[1][:-1], bands[1][1:])], bands[0].tolist())))
            ws = np.bincount(k, minlength=KA.shape[1])
            print("   per winding:", {int(i): int(c) for i, c in enumerate(ws) if c})
            o = np.argsort(-(KB - KA)[a, k] if "NEW" in lab else -(KA - KB)[a, k])[:15]
            for i in o:
                print(f"     z{z[a[i]]:.0f} w{k[i]}  old {KA[a[i], k[i]]:5.1f}  new {KB[a[i], k[i]]:5.1f}")


if __name__ == "__main__":
    main()
