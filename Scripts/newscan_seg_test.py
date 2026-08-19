"""Does the existing nnU-Net (Dataset502) transfer to the NEW scan?

The new reconstruction is a SEPARATE ACQUISITION of the same scroll (user: "almost
as if scanned by a different x-ray"), so there are TWO possible domain shifts:
  * SCALE    -- film layer is ~22px here vs the ~18px the model trained on
                (the new scan is ~1.2508x finer in-plane)
  * CONTRAST -- different beam/detector/noise and edge response
Testing both variants separates them:
  native  -- predict at full new-scan resolution        (scale + contrast shift)
  scaled  -- predict after downsampling by 1.2508       (contrast shift only)
If native fails and scaled works, segment at old scale and upsample the mask.

Modes:
  chunks -- build a 20-slice volume_*.h5 (key 'volume') from the per-slice files,
            optionally downsampled, ready for hdf5_to_nifti_for_predict.py
  check  -- overlay the predicted labels on the CT + print geometry stats to
            compare against the OLD scan's measured values:
              film (class>=1) 18.0px, emulsion (class==2) 4.0px, spacing 21.0px
            times 1.2508 for the native variant -> 22.5 / 5.0 / 26.3 px.

Usage:
  python Scripts/newscan_seg_test.py chunks --in-dir 01_Mickey_sprockets \
      --z0 500 --z1 519 --out-dir <dir> [--scale 1.2508]
  python Scripts/newscan_seg_test.py check --pred <labels.nii.gz> \
      --vol <volume_*.h5> --out-dir <dir> --tag native
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py


def cmd_chunks(args):
    zs = list(range(args.z0, args.z1 + 1))
    sl = []
    for z in zs:
        p = os.path.join(args.in_dir, f"Mickey_merged_{z:04d}.h5")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        with h5py.File(p, "r") as f:
            a = np.asarray(f["image"])
        sl.append(a)
    vol = np.stack(sl)
    print(f"stacked {vol.shape} from z{args.z0}-{args.z1}")
    if args.scale and abs(args.scale - 1.0) > 1e-6:
        from scipy.ndimage import gaussian_filter, zoom
        s = args.scale
        sigma = 0.5 * np.sqrt(max(s * s - 1.0, 0.0))     # anti-alias before decimate
        out = []
        for i in range(vol.shape[0]):
            out.append(zoom(gaussian_filter(vol[i].astype(np.float32), sigma),
                            1.0 / s, order=1))
        vol = np.stack(out)
        print(f"  downsampled by {s} (anti-aliased) -> {vol.shape}")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"volume_{args.z0:04d}-{args.z1:04d}.h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("volume", data=vol.astype(np.uint16))
    print(f"wrote {out}")


def ray_stats(seg2, n_rays=180):
    """Median run length of film (class>=1) and emulsion (class==2) along radial
    rays, plus emulsion-to-emulsion spacing -- the same measurement that gave
    18.0 / 4.0 / 21.0 px on the old scan."""
    from scipy.ndimage import label
    ys, xs = np.nonzero(seg2 > 0)
    if ys.size < 1000:
        return None
    cy, cx = ys.mean(), xs.mean()
    r = np.hypot(ys - cy, xs - cx)
    r0, r1 = np.percentile(r, [1, 99])
    film, emul, spac = [], [], []
    from scipy.ndimage import map_coordinates
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        rr = np.arange(r0, r1)
        v = map_coordinates(seg2.astype(np.int16),
                            [cy + rr * np.sin(th), cx + rr * np.cos(th)], order=0)
        for cls, acc in ((1, film), (2, emul)):
            m = (v >= 1) if cls == 1 else (v == 2)
            lab, n = label(m)
            if n:
                acc += [int((lab == i).sum()) for i in range(1, n + 1)]
        lab, n = label(v == 2)
        if n >= 2:
            c = [rr[lab == i].mean() for i in range(1, n + 1)]
            spac += np.diff(sorted(c)).tolist()
    f = lambda a: (float(np.median(a)) if a else float("nan"))
    sp = [s for s in spac if 5 < s < 60]
    return dict(film=f(film), emul=f(emul), spacing=f(sp),
                n_film=len(film), n_emul=len(emul))


def cmd_check(args):
    import nibabel as nib
    seg = np.asarray(nib.load(args.pred).dataobj)
    with h5py.File(args.vol, "r") as f:
        vol = np.asarray(f["volume"], np.float32)
    if seg.shape != vol.shape:
        print(f"NOTE seg {seg.shape} vs vol {vol.shape}")
        if seg.shape == vol.shape[::-1] or seg.shape[1:] == vol.shape[1:][::-1]:
            print("  ^ AXIS MISMATCH (the old H<->W transpose trap) -- "
                  "orienting seg to the volume")
            seg = np.swapaxes(seg, 1, 2)
    os.makedirs(args.out_dir, exist_ok=True)
    frac = [float((seg == c).mean()) for c in (0, 1, 2)]
    print(f"[{args.tag}] shape {seg.shape}  class fractions "
          f"bg {frac[0]:.3f} base {frac[1]:.3f} emul {frac[2]:.3f}")
    mid = seg.shape[0] // 2
    st = ray_stats(seg[mid])
    if st:
        print(f"[{args.tag}] mid-slice thickness: film {st['film']:.1f}px "
              f"emulsion {st['emul']:.1f}px spacing {st['spacing']:.1f}px "
              f"(runs {st['n_film']}/{st['n_emul']})")
        exp = (18.0, 4.0, 21.0) if args.tag == "scaled" else (22.5, 5.0, 26.3)
        print(f"[{args.tag}] EXPECTED for a good transfer: film {exp[0]:.1f} "
              f"emulsion {exp[1]:.1f} spacing {exp[2]:.1f} px")

    ci = np.array([[0, 0, 0], [0.15, 0.5, 1.0], [1.0, 0.2, 0.2]])
    for k in (0, seg.shape[0] // 2):
        img = vol[k]
        lo, hi = np.percentile(img, [1, 99.5])
        g = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = np.repeat(g[..., None], 3, axis=2)
        ov = rgb.copy()
        for c in (1, 2):
            m = seg[k] == c
            ov[m] = 0.45 * rgb[m] + 0.55 * ci[c]
        ys, xs = np.nonzero(seg[k] > 0)
        cy, cx = (ys.mean(), xs.mean()) if ys.size else (img.shape[0] / 2,
                                                        img.shape[1] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        ax[0].imshow(ov); ax[0].set_title(f"{args.tag} z-index {k}: full slice")
        y0 = int(cy) - 60; x0 = int(cx) + 700
        ax[1].imshow(ov[max(0, y0):y0 + 320, x0:x0 + 320], interpolation="nearest")
        ax[1].set_title("zoom (blue=base, red=emulsion) -- bands must be "
                        "continuous + emulsion a thin line")
        for a_ in ax:
            a_.set_xticks([]); a_.set_yticks([])
        plt.tight_layout()
        p = os.path.join(args.out_dir, f"segcheck_{args.tag}_{k:02d}.png")
        plt.savefig(p, dpi=110); plt.close()
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("chunks")
    c.add_argument("--in-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--z0", type=int, required=True)
    c.add_argument("--z1", type=int, required=True)
    c.add_argument("--scale", type=float, default=1.0)
    k = sub.add_parser("check")
    k.add_argument("--pred", required=True)
    k.add_argument("--vol", required=True)
    k.add_argument("--out-dir", required=True)
    k.add_argument("--tag", default="native")
    args = ap.parse_args()
    (cmd_chunks if args.mode == "chunks" else cmd_check)(args)


if __name__ == "__main__":
    main()
