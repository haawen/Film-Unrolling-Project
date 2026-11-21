"""Can a CONTENT-FREE neighbour of the emulsion serve as the stripe background?

The video's faint vertical stripes are fixed-z bands in the strip that come from
the volume reconstruction/stitching (not from the unroll). They zigzag about once
per winding, i.e. they are a SURFACE in the CT volume, not a pure function of z,
so a single global row profile cannot remove them.

The film base (and the air gap) a few px off the emulsion carries no picture but
sits at almost the same place in the volume, so it should carry the same stripes.
This samples, for a subset of kept windings over the picture band, the CT along
the delivered path (offset 0, same transverse average as the render) and at
several offsets along the path NORMAL, and saves them (float16) so the local
analysis can test: does the z-high-pass of an offset strip predict the z-high-
pass of the emulsion strip?

Geometry = the deliverable's (matched_walks.npz from v13 --inspect-anchors with
the render flags), z-interpolated exactly like the render (`_z_map`, hold).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unwrapping.inr.unroll_continuous import load_ct, norm_slice, discover_ct_slices
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_walk_wholeroll_v13 import _z_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windings", default="4,9,14,19,24,29")
    ap.add_argument("--offsets", default="-12,-9,-6,0,6,9,12")
    ap.add_argument("--z-lo", type=int, default=500)
    ap.add_argument("--z-hi", type=int, default=1950)
    ap.add_argument("--transverse-px", type=int, default=3)
    args = ap.parse_args()

    d = np.load(args.matched, allow_pickle=True)
    z_anchor = np.asarray(d["z_anchor"], float)
    cx = np.asarray(d["cx_a"], float); cy = np.asarray(d["cy_a"], float)
    P = d["paths"]
    ws = [int(v) for v in args.windings.split(",")]
    offs = [float(v) for v in args.offsets.split(",")]
    maps, Ks = [], []
    for k in ws:
        X = np.stack([np.asarray(P[a][k], float)[:, 0] for a in range(len(z_anchor))])
        Y = np.stack([np.asarray(P[a][k], float)[:, 1] for a in range(len(z_anchor))])
        maps.append((_z_map(z_anchor, X), _z_map(z_anchor, Y))); Ks.append(X.shape[1])
    starts = np.cumsum([0] + Ks); C = int(starts[-1])
    z_to_path = discover_ct_slices(args.ct_dir)
    zs = [z for z in sorted(z_to_path) if args.z_lo <= z <= args.z_hi]
    out = np.zeros((len(offs), len(zs), C), np.float16)
    tw = max(1, args.transverse_px)
    taps = np.linspace(-(tw - 1) / 2.0, (tw - 1) / 2.0, tw)
    # radial sign of the normal, per winding: +1 if the normal points OUTWARD
    sign = []
    for j, z in enumerate(zs):
        img = norm_slice(load_ct(z_to_path[z]))
        a_near = int(np.argmin(np.abs(z_anchor - z)))
        for i, (fX, fY) in enumerate(maps):
            xs, ys = fX(z), fY(z)
            dx, dy = np.gradient(xs), np.gradient(ys)
            L = np.hypot(dx, dy) + 1e-9
            nx, ny = -dy / L, dx / L
            if j == 0:
                rx, ry = xs - cx[a_near], ys - cy[a_near]
                sign.append(float(np.sign(np.median(nx * rx + ny * ry))))
            s = sign[i]                              # offsets are OUTWARD-positive
            for o, off in enumerate(offs):
                acc = np.zeros_like(xs)
                tl = taps if off == 0 else [0.0]
                for t in tl:
                    acc += bilinear_sample(img, xs + s * (off + t) * nx,
                                           ys + s * (off + t) * ny)
                out[o, j, starts[i]:starts[i + 1]] = acc / len(tl)
        if j % 100 == 0:
            print(f"  [{j + 1}/{len(zs)}] z{z}", flush=True)
    np.savez(args.out, strips=out, offsets=np.array(offs), z=np.array(zs),
             windings=np.array(ws), starts=starts)
    print(f"wrote {args.out}: {out.shape} (offsets x z x cols), windings {ws}, "
          f"normal sign (outward=+1) {sign}")


if __name__ == "__main__":
    main()
