"""Ten evenly-spaced Roll2 slices: CT | segmentation | emulsion walk.

Roll2 only ever had 4 segmented slices (z 200/700/1200/1700), and two of those
walked poorly, so the 4-anchor z-interpolation left the rough strip smeared.
This produces TEN evenly-spaced anchors across the good film band and shows,
per slice, the three panels needed to judge whether a 10-anchor interpolation
would give a decent strip: the raw CT, the segmentation, and the walk.

CT and segmentation are BOTH read from the same 20-slice chunk at the same
local index, so they are z-aligned by construction (no tiff/chunk off-by-one).

ORIENTATION IS CHECKED PER SLICE. `nnunet_softmax_to_h5_probs.py` only fixes an
H<->W transpose, and only when H != W -- Roll2 is 3064x3064 (square), so that
guard is a structural no-op here, and the delivered `data/roll2_probs/` maps
turned out to be VERTICALLY FLIPPED against the CT. That flip was invisible in
the seg-only overlays (the walk lives in seg coordinates) and only showed up
once the CT was sampled along the walk. So here every slice is tested against
its own CT by FFT cross-correlation over the 8 dihedral orientations; the one
that peaks at zero shift wins, and the chosen transform is printed and drawn.

Usage (Merlin):
  python -m unwrapping.inr.roll2_ten_slices \
      --chunk-dir 02_Sample_3d --probs-dir data/roll2_probs10 \
      --out-dir unwrapping/inr/results/roll2_ten
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.walk_emulsion import (
    seed_run_intervals, film_seed_azimuth, walk, dedup_paths,
    recenter_on_emulsion, smooth_path, TWO_PI,
)

ROLL1_SPACING = 21.0

# the 8 dihedral orientations a seg can land in relative to its CT
ORIENTS = {
    "identity":  lambda a: a,
    "flipV":     lambda a: a[::-1],
    "flipH":     lambda a: a[:, ::-1],
    "rot180":    lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,
    "T+flipV":   lambda a: a.T[::-1],
    "T+flipH":   lambda a: a.T[:, ::-1],
    "rot90":     lambda a: np.rot90(a),
}


def chunk_name(z0, chunk_dir):
    for z1 in (z0 + 19, z0 + 15, z0 + 3):
        p = os.path.join(chunk_dir, "volume_%04d-%04d.h5" % (z0, z1))
        if os.path.exists(p):
            return p, z1
    raise FileNotFoundError("no CT chunk starting at z0=%d in %s" % (z0, chunk_dir))


def load_ct(z0, idx, chunk_dir):
    ctp, z1 = chunk_name(z0, chunk_dir)
    with h5py.File(ctp, "r") as f:
        key = "volume" if "volume" in f else list(f.keys())[0]
        return f[key][idx].astype(np.float32), z1


def load_pair(z0, idx, chunk_dir, probs_dir=None, label_dir=None, prefix="Roll2Ten"):
    """CT slice + its segmentation, both keyed to the SAME chunk + local index.

    Two seg sources: a 3D softmax chunk (`probs_dir`, the path the original 4
    Roll2 anchors took) or a 2D per-slice label tif (`label_dir`, nnU-Net 2D
    Dataset503 -- ~60x cheaper because a 2D net needs no z-context).
    """
    ct, z1 = load_ct(z0, idx, chunk_dir)
    z = z0 + idx
    if label_dir:
        import tifffile
        lp = os.path.join(label_dir, "%s_%04d.tif" % (prefix, z))
        seg = np.asarray(tifffile.imread(lp)).astype(np.uint8)
        if seg.ndim == 3:                       # (1,H,W) or (H,W,1)
            seg = seg.squeeze()
    else:
        pp = os.path.join(probs_dir,
                          "volume_%04d-%04d_Probabilities.h5" % (z0, z1))
        with h5py.File(pp, "r") as f:
            probs = f["exported_data"][idx]     # (H, W, 3)
        seg = np.argmax(probs, axis=-1).astype(np.uint8)
    return ct, seg, z


def pick_orientation(ct, seg, k=8):
    """FFT xcorr of the film mask against the CT over the 8 dihedral maps.

    The right orientation both correlates best AND needs no translation to do
    it; a wrong one on a near-radially-symmetric roll still correlates highly
    but only after a shift, which is exactly how the V-flip stayed hidden.
    Score = peak correlation penalised by the shift it needed.
    """
    H = (ct.shape[0] // k) * k
    ds = lambda a: a[:H, :H].reshape(H // k, k, H // k, k).mean(axis=(1, 3))
    C = ds(ct)
    C = (C - C.mean()) / (C.std() + 1e-9)
    rows = []
    for name, fn in ORIENTS.items():
        S = fn((seg >= 1).astype(np.float32))
        if S.shape != seg.shape:
            continue
        S = ds(S)
        S = (S - S.mean()) / (S.std() + 1e-9)
        F = np.fft.ifft2(np.fft.fft2(C) * np.conj(np.fft.fft2(S))).real
        p = np.unravel_index(np.argmax(F), F.shape)
        n0, n1 = F.shape
        sh = (((p[0] + n0 // 2) % n0 - n0 // 2) * k,
              ((p[1] + n1 // 2) % n1 - n1 // 2) * k)
        corr = float(F.max() / C.size)
        shift = float(math.hypot(sh[0], sh[1]))
        rows.append((corr - shift / 400.0, corr, shift, name, sh))
    rows.sort(reverse=True)
    return rows[0], rows


def measure_spacing(emul, cx, cy, r0, r1):
    best_d, best_n = None, -1
    for az in np.linspace(0, TWO_PI, 360, endpoint=False):
        runs = seed_run_intervals(emul, cx, cy, az, r0, r1, min_thick=1.5)
        if len(runs) > best_n:
            best_n = len(runs)
            mids = np.sort([0.5 * (a + b) for a, b in runs])
            best_d = np.diff(mids) if mids.size > 1 else None
    if best_d is not None and best_d.size:
        return float(np.median(best_d)), best_n
    return float("nan"), best_n


def walk_slice(seg):
    emul = (seg == 2).astype(np.float32)
    film = (seg >= 1)
    cy, cx = find_spool_center(seg)
    ys, xs = np.where(film)
    rf = np.hypot(ys - cy, xs - cx)
    r0, r1 = rf.min() + 5, rf.max() - 5
    seam = _detect_seam_angle(film, (cy, cx), rf.min(), rf.max())
    sp, nruns = measure_spacing(emul, cx, cy, r0, r1)
    scale = sp / ROLL1_SPACING if np.isfinite(sp) and sp > 0 else 1.0

    step = max(1.0, 2.0 * scale)
    snap_max = max(3.0, 8.0 * scale)
    snap_accept = max(2.5, 6.0 * scale)
    jump_max = max(2.0, 4.0 * scale)
    max_dr = max(1.0, 2.0 * scale)
    min_sep = max(3.0, 6.0 * scale)
    film_mt = max(2.5, 6.0 * scale)
    rec_search = max(2.0, 6.0 * scale)

    seed_az, seeds = film_seed_azimuth(film.astype(np.float32), emul, cx, cy,
                                       r0, r1, film_mt, snap_max, infill=True,
                                       verbose=False)
    stop_extent = math.pi - math.radians(6.0)
    tan = np.array([-math.sin(seed_az), math.cos(seed_az)])
    paths = []
    for rs in seeds:
        p0 = np.array([cx + rs * math.cos(seed_az), cy + rs * math.sin(seed_az)])
        f = walk(emul, p0, tan, cx, cy, step, snap_max, 40, stop_extent,
                 jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        b = walk(emul, p0, -tan, cx, cy, step, snap_max, 40, stop_extent,
                 jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        pth = np.vstack([b[::-1], f[1:]])
        pth = smooth_path(pth, 0.6)
        pth = recenter_on_emulsion(pth, emul, rec_search, 20.0)
        paths.append(pth)
    paths = dedup_paths(paths, cx, cy, min_sep)
    return dict(paths=paths, cx=float(cx), cy=float(cy), spacing=sp,
                scale=scale, n_seed=len(seeds), n_kept=len(paths),
                seam=math.degrees(seam), r0=float(r0), r1=float(r1))


def panel(z, ct, seg, w, orient, out_png):
    lo, hi = np.percentile(ct, [1, 99])
    disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.zeros(seg.shape + (3,), np.float32)
    rgb[seg == 1] = (0.50, 0.50, 0.50)
    rgb[seg == 2] = (1.0, 0.15, 0.15)
    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(1, 3, figsize=(30, 10.4))
    ax[0].imshow(disp, cmap="gray")
    ax[0].set_title("CT  z=%d" % z, fontsize=15)
    ax[1].imshow(rgb)
    ax[1].set_title("SEG  (grey=base, red=emulsion)   orient=%s" % orient,
                    fontsize=15)
    ax[2].imshow(disp, cmap="gray")
    for k, p in enumerate(w["paths"]):
        ax[2].plot(p[:, 0], p[:, 1], "-", lw=0.7,
                   color=cmap(k / max(1, w["n_kept"])))
    ax[2].plot(w["cx"], w["cy"], "c+", ms=12)
    ax[2].set_title("WALK on CT  %d windings  (spacing %.1fpx, scale %.2f)"
                    % (w["n_kept"], w["spacing"], w["scale"]), fontsize=15)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=92)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--chunk-dir", default="02_Sample_3d")
    ap.add_argument("--probs-dir", default=None,
                    help="Dir of 3D softmax chunks (volume_*_Probabilities.h5).")
    ap.add_argument("--label-dir", default=None,
                    help="Dir of 2D nnU-Net label tifs (<prefix>_<z>.tif).")
    ap.add_argument("--prefix", default="Roll2Ten")
    ap.add_argument("--chunk-z0s",
                    default="200,360,540,700,880,1060,1220,1400,1580,1740",
                    help="Chunk start z of each anchor (evenly spaced).")
    ap.add_argument("--idx", type=int, default=10,
                    help="Slice index within the 20-slice chunk.")
    ap.add_argument("--out-dir",
                    default=os.path.join(proj,
                                         "unwrapping/inr/results/roll2_ten"))
    args = ap.parse_args()
    if not args.probs_dir and not args.label_dir:
        raise SystemExit("need --probs-dir or --label-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    z0s = [int(x) for x in args.chunk_z0s.split(",")]

    pngs, rows = [], []
    for z0 in z0s:
        ct, seg, z = load_pair(z0, args.idx, args.chunk_dir,
                               args.probs_dir, args.label_dir, args.prefix)
        best, allr = pick_orientation(ct, seg)
        _, corr, shift, name, sh = best
        seg = ORIENTS[name](seg)
        nxt = allr[1]
        print("\n=== z=%d (chunk %d) ===" % (z, z0), flush=True)
        print("  orient=%s  corr=%.3f shift=%s   (next: %s corr=%.3f shift=%.0fpx)"
              % (name, corr, sh, nxt[3], nxt[1], nxt[2]), flush=True)
        w = walk_slice(seg)
        print("  spacing=%.1f scale=%.3f seeds=%d kept=%d r[%.0f,%.0f] seam~%.0fdeg"
              % (w["spacing"], w["scale"], w["n_seed"], w["n_kept"],
                 w["r0"], w["r1"], w["seam"]), flush=True)
        p = os.path.join(args.out_dir, "slice_z%04d.png" % z)
        panel(z, ct, seg, w, name, p)
        pngs.append(p)
        rows.append((z, name, w))
        store = {("w%d" % i): pp for i, pp in enumerate(w["paths"])}
        store.update(cx=w["cx"], cy=w["cy"], spacing=w["spacing"],
                     seam=w["seam"], z=z)
        np.savez_compressed(
            os.path.join(args.out_dir, "walk_z%04d.npz" % z), **store)

    ims = [Image.open(p) for p in pngs]
    wid = 1500
    ims = [im.resize((wid, int(im.size[1] * wid / im.size[0]))) for im in ims]
    hh = ims[0].size[1]
    canv = Image.new("RGB", (wid, hh * len(ims)), "white")
    for i, im in enumerate(ims):
        canv.paste(im, (0, i * hh))
    canv.save(os.path.join(args.out_dir, "ALL_ten_slices.png"))

    print("\n==== SUMMARY ====")
    print("%6s %10s %8s %5s %5s %7s  r-range"
          % ("z", "orient", "spacing", "seed", "kept", "seam"))
    for z, name, w in rows:
        print("%6d %10s %8.1f %5d %5d %7.0f  [%.0f,%.0f]"
              % (z, name, w["spacing"], w["n_seed"], w["n_kept"], w["seam"],
                 w["r0"], w["r1"]))
    print("\n-> %s/ALL_ten_slices.png" % args.out_dir)


if __name__ == "__main__":
    main()
