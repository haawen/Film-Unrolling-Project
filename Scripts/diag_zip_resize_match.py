"""Test the premise that the OLD stitch is simply the NEW reconstruction RESIZED
(down to the segmentation model's input resolution), possibly with an axis flip.

If that is true the two slices are the SAME IMAGE at different scale, so:
  downsample the zip by s (with an anti-alias prefilter), apply each of the 8
  square symmetries (plain array ops), cross-correlate against the old slice.
The correct combination should give a large, unambiguous normalised-correlation
peak; wrong ones ~0. This also returns the translation for free (the peak position)
and, by sweeping s, the scale.

This is the test my FIRST attempt botched: it decimated by 4 with no prefilter,
which aliased the 21/26px ring texture into noise (ncc 0.06-0.09 for everything).
Anti-aliasing is the whole difference. Everything since -- boundary profiles, polar
correlation, curve-template alignment, seam detection, the 8-way dihedral scan --
was working around that mistake, and none of it needed to exist if the premise holds.

Reference points already established (all rotation-independent, so still valid):
scale ~1.245-1.251 (film ring radii, winding spacing, curve alignment agree to
0.15%); old film ring r623-1347, zip r776-1683.

Usage:
  python Scripts/diag_zip_resize_match.py --old-dir 01_Mickey_hdf \
      --zip-dir 01_Mickey_sprockets --out-dir <dir> \
      --pairs 848:490 863:510 1988:1940 2003:1960
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, zoom

TRANSFORMS = {
    "identity":       lambda a: a,
    "rot90":          lambda a: np.rot90(a, 1),
    "rot180":         lambda a: np.rot90(a, 2),
    "rot270":         lambda a: np.rot90(a, 3),
    "transpose":      lambda a: a.T,
    "anti_transpose": lambda a: np.rot90(a, 2).T,
    "flip_x":         lambda a: a[:, ::-1],
    "flip_y":         lambda a: a[::-1, :],
}


def prep(a, hp=12.0):
    """High-pass, zero-mean, unit-norm. The high-pass removes the two frames'
    different background/offset levels (the old one is a constant 12 outside the
    reconstruction circle) and keeps the ring structure that carries the match."""
    b = a.astype(np.float32)
    b = b - gaussian_filter(b, hp)
    b -= b.mean()
    n = np.linalg.norm(b)
    return b / n if n > 0 else b


def centre_crop(a, half, cy=None, cx=None):
    cy = a.shape[0] / 2 if cy is None else cy
    cx = a.shape[1] / 2 if cx is None else cx
    y0 = int(round(cy - half)); x0 = int(round(cx - half))
    y0 = max(0, min(y0, a.shape[0] - 2 * half))
    x0 = max(0, min(x0, a.shape[1] - 2 * half))
    return a[y0:y0 + 2 * half, x0:x0 + 2 * half]


def ncc_peak(A, B):
    """Peak circular cross-correlation of two unit-norm images + the shift."""
    n = max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1])
    Ap = np.zeros(n, np.float32); Ap[:A.shape[0], :A.shape[1]] = A
    Bp = np.zeros(n, np.float32); Bp[:B.shape[0], :B.shape[1]] = B
    cc = np.fft.irfft2(np.fft.rfft2(Ap) * np.conj(np.fft.rfft2(Bp)), s=n)
    k = int(np.argmax(cc))
    dy, dx = np.unravel_index(k, cc.shape)
    if dy > n[0] // 2:
        dy -= n[0]
    if dx > n[1] // 2:
        dx -= n[1]
    return float(cc.flat[k]), int(dy), int(dx)


def rot_scale_sweep(O, Z, half, work=800, mirrors=(False, True),
                    step_deg=1.0, scale=1.2508):
    """Coarse mirror x rotation sweep by raw-pixel NCC, at reduced resolution.

    The two files are separate RECONSTRUCTIONS of the same scan (old
    "Stitch_Export" at 3064^2, new "merged_RIGID_nobin" at 3738x3769) -- the name
    says a rigid transform was applied when merging, so a real rotation is
    expected, and only the MIRROR is discrete.

    Both images are prefiltered before decimation. That prefilter is the whole
    reason this can work: the first attempt at this test decimated by 4 with no
    filter, aliasing the 21/26px ring texture into noise (ncc 0.06-0.09 for every
    candidate), which sent me off into four indirect estimators.
    """
    from scipy.ndimage import rotate
    A0 = centre_crop(O, half)
    Z0 = centre_crop(downsample(Z, scale), half)
    fo = A0.shape[0] / work
    A = prep(downsample(A0, fo), hp=6.0)
    Zw = downsample(Z0, fo)
    out = []
    for mir in mirrors:
        Zm = Zw[:, ::-1] if mir else Zw
        for th in np.arange(0.0, 360.0, step_deg):
            B = prep(rotate(Zm, th, order=1, reshape=False), hp=6.0)
            v, dy, dx = ncc_peak(A, B)
            out.append((v, bool(mir), float(th), dy * fo, dx * fo))
    out.sort(key=lambda t: -t[0])
    return out


def downsample(a, s):
    """Downsample by factor s WITH an anti-alias prefilter (the load-bearing bit)."""
    if abs(s - 1.0) < 1e-6:
        return a.astype(np.float32)
    sigma = 0.5 * np.sqrt(max(s * s - 1.0, 0.0))
    return zoom(gaussian_filter(a.astype(np.float32), sigma), 1.0 / s, order=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="old_z:zip_z pairs, e.g. 848:490")
    ap.add_argument("--scale", type=float, default=1.2508)
    ap.add_argument("--scale-sweep", type=float, nargs=3,
                    default=[1.230, 1.270, 0.002],
                    help="lo hi step for the winner's scale refinement")
    ap.add_argument("--rot-sweep", action="store_true",
                    help="run the mirror x rotation NCC sweep first")
    ap.add_argument("--half", type=int, default=1350,
                    help="half-size of the compared centre crop (old px)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    oldmap = {}
    for p in glob.glob(os.path.join(args.old_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            oldmap[int(m.group(1))] = p

    # ── mirror x rotation sweep first: the 8 axis conventions are just 8 points of
    #    this sweep, and "merged_rigid" says the true angle need not be one of them ──
    if args.rot_sweep:
        for pair in args.pairs[:2]:
            zo, zz = (int(x) for x in pair.split(":"))
            if zo not in oldmap:
                continue
            with h5py.File(oldmap[zo], "r") as f:
                O = np.asarray(f["image"], np.float32)
            with h5py.File(os.path.join(args.zip_dir,
                                        f"Mickey_merged_{zz:04d}.h5"), "r") as f:
                Z = np.asarray(f["image"], np.float32)
            res = rot_scale_sweep(O, Z, args.half, scale=args.scale)
            print(f"\n=== ROTATION SWEEP old z{zo} vs zip z{zz} "
                  f"(s={args.scale}) -- top 8 of {len(res)} ===", flush=True)
            for v, mir, th, dy, dx in res[:8]:
                print(f"   ncc {v:.4f}  mirror={str(mir):5s} theta {th:6.1f}  "
                      f"shift {dy:+8.1f},{dx:+8.1f}")
            v0 = res[0][0]; vmed = np.median([r[0] for r in res])
            print(f"   peak {v0:.4f} vs median {vmed:.4f} = "
                  f"{v0 / max(vmed, 1e-9):.1f}x  <- a real match should be MANY x "
                  f"the median; ~1x means no match at any angle", flush=True)

    all_rows = []
    for pair in args.pairs:
        zo, zz = (int(x) for x in pair.split(":"))
        if zo not in oldmap:
            print(f"old z{zo} missing"); continue
        with h5py.File(oldmap[zo], "r") as f:
            O = np.asarray(f["image"], np.float32)
        zp = os.path.join(args.zip_dir, f"Mickey_merged_{zz:04d}.h5")
        with h5py.File(zp, "r") as f:
            Z = np.asarray(f["image"], np.float32)

        A = prep(centre_crop(O, args.half))
        Zs = downsample(Z, args.scale)
        print(f"\n=== old z{zo} {O.shape} vs zip z{zz} {Z.shape} "
              f"-> resized {Zs.shape} at s={args.scale} ===", flush=True)
        print(f"{'transform':>15} {'peak NCC':>10} {'shift dy,dx':>14}")
        rows = []
        for name, fn in TRANSFORMS.items():
            B = prep(centre_crop(fn(Zs), args.half))
            v, dy, dx = ncc_peak(A, B)
            rows.append(dict(old_z=zo, zip_z=zz, name=name, ncc=v, dy=dy, dx=dx))
            print(f"{name:>15} {v:10.4f} {dy:7d},{dx:6d}", flush=True)
        rows.sort(key=lambda r: -r["ncc"])
        print(f"  -> best {rows[0]['name']} ncc {rows[0]['ncc']:.4f}; "
              f"runner-up {rows[1]['name']} {rows[1]['ncc']:.4f} "
              f"(ratio {rows[0]['ncc'] / max(rows[1]['ncc'], 1e-9):.2f}x)")
        all_rows += rows

    if not all_rows:
        return
    print("\n=== VERDICT: the true convention must win at EVERY pair ===")
    for name in TRANSFORMS:
        v = [r["ncc"] for r in all_rows if r["name"] == name]
        print(f"  {name:>15}: ncc {np.round(v, 4).tolist()} mean {np.mean(v):.4f}")
    best_name = max(TRANSFORMS,
                    key=lambda n: np.mean([r["ncc"] for r in all_rows
                                           if r["name"] == n]))
    print(f"  winner: {best_name}")

    # ── refine the scale for the winner on the first pair ──
    zo, zz = (int(x) for x in args.pairs[0].split(":"))
    with h5py.File(oldmap[zo], "r") as f:
        O = np.asarray(f["image"], np.float32)
    with h5py.File(os.path.join(args.zip_dir, f"Mickey_merged_{zz:04d}.h5"), "r") as f:
        Z = np.asarray(f["image"], np.float32)
    A = prep(centre_crop(O, args.half))
    lo, hi, st = args.scale_sweep
    curve = []
    for s in np.arange(lo, hi + 1e-9, st):
        B = prep(centre_crop(TRANSFORMS[best_name](downsample(Z, s)), args.half))
        v, dy, dx = ncc_peak(A, B)
        curve.append((float(s), v, dy, dx))
        print(f"  s {s:.4f}: ncc {v:.4f} shift {dy},{dx}", flush=True)
    s_best, v_best, dy_b, dx_b = max(curve, key=lambda t: t[1])
    print(f"\n  SCALE for {best_name}: {s_best:.4f} (ncc {v_best:.4f}), "
          f"shift dy {dy_b} dx {dx_b}")

    ss = [c[0] for c in curve]; vv = [c[1] for c in curve]
    plt.figure(figsize=(8, 4))
    plt.plot(ss, vv, "o-")
    plt.axvline(s_best, color="r", ls="--")
    plt.xlabel("scale"); plt.ylabel("peak NCC"); plt.grid(alpha=0.3)
    plt.title(f"{best_name}: NCC vs scale (old z{zo} vs zip z{zz})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scale_sweep.png"), dpi=130)

    with open(os.path.join(args.out_dir, "resize_match.json"), "w") as f:
        json.dump(dict(rows=all_rows, winner=best_name, scale=s_best,
                       ncc=v_best, shift=[dy_b, dx_b], curve=curve), f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
