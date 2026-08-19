"""Localise geometry tears in a rendered strip WITHOUT knowing where to look.

A geometry error (winding jump, bad spool centre, mis-tracked winding) displaces
one strip ROW relative to its z-neighbours. Film content does not do that: the
emulsion is continuous across the film's width, so |strip[z+1] - strip[z]| is
small everywhere the geometry is right, whatever the picture shows. Averaging
that over the whole z axis -- which is what diag_strip_artifact's column profile
does -- dilutes a defect confined to a z band into nothing (measured: the 7 s
artifact does not clear 1.4x the median that way). So resolve it in BOTH axes and
normalise each z-block against its own median, which removes the fact that some
z bands are intrinsically busier than others.

Outputs a PNG of the map (bright = torn) plus the worst blocks as text, in strip
coordinates AND in video coordinates (cell, frame, video x) so a finding can be
checked against what was actually seen.

Usage:
  python Scripts/diag_zstep_map.py --strip .../wholeroll.npy --out map.png \
      --pitch 1130.14 --n-frames 229
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rblock", type=int, default=8)
    ap.add_argument("--cblock", type=int, default=64)
    ap.add_argument("--pitch", type=float, default=1130.14)
    ap.add_argument("--n-frames", type=int, default=229)
    ap.add_argument("--z-first", type=int, default=40)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=200,
                    help="Rows loaded at a time (the strip is memmapped).")
    args = ap.parse_args()

    A = np.load(args.strip, mmap_mode="r")
    Z, C = A.shape
    nr, nc = Z // args.rblock, C // args.cblock
    print(f"{args.strip} {A.shape} -> map {nr} x {nc} "
          f"({args.rblock} rows x {args.cblock} cols per block)")

    M = np.zeros((nr, nc), np.float32)
    for r0 in range(0, nr * args.rblock, args.chunk * args.rblock):
        r1 = min(r0 + args.chunk * args.rblock, nr * args.rblock)
        blk = np.asarray(A[r0:min(r1 + 1, Z)], dtype=np.float32)
        d = np.abs(np.diff(blk, axis=0))          # step along z
        d = d[:r1 - r0, :nc * args.cblock]         # trim the ragged last column block
        d = d[:(len(d) // args.rblock) * args.rblock]
        d = d.reshape(-1, args.rblock, nc, args.cblock).mean(axis=(1, 3))
        M[r0 // args.rblock:r0 // args.rblock + len(d)] = d

    med = np.median(M, axis=1, keepdims=True)
    R = M / np.maximum(med, 1e-6)                 # per-z-band normalisation

    import cv2
    v = np.clip((R - 1.0) / 1.5, 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for f in range(0, args.n_frames, 10):
        x = int((args.n_frames - 1 - f) * args.pitch / args.cblock)
        if 0 <= x < nc:
            cv2.line(img, (x, 0), (x, 12), (255, 255, 255), 1)
            cv2.putText(img, f"f{f}", (x + 2, 24), 0, 0.35, (255, 255, 255), 1)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out} {img.shape}")

    print(f"\nworst {args.top} blocks (ratio to that z-band's median step):")
    flat = np.argsort(-R, axis=None)[:args.top]
    for k in flat:
        i, j = np.unravel_index(k, R.shape)
        z = i * args.rblock + args.z_first
        c = j * args.cblock
        cell = c / args.pitch
        frame = args.n_frames - 1 - cell
        vx = (z - args.z_first) * 1504.0 / (Z - 1)
        print(f"  z{z:5d}  col {c:7d}  ratio {R[i,j]:6.2f}   "
              f"= video t {frame/25:5.2f}s (frame {frame:6.1f}), "
              f"video x {vx:6.0f}/1504")

    print("\nrow profile (max over columns), worst 12 z-bands:")
    rp = R.max(axis=1)
    for i in np.argsort(-rp)[:12]:
        z = i * args.rblock + args.z_first
        vx = (z - args.z_first) * 1504.0 / (Z - 1)
        print(f"  z{z:5d} (video x {vx:4.0f})  max ratio {rp[i]:6.2f}")


if __name__ == "__main__":
    main()
