"""First-look geometry of a NEW CT volume (high-res Mickey scan, 2026-11), from the
raw CT only (no segmentation yet): roll centre, radial film extent, film-layer
count and winding pitch per slice. These numbers set the scale factor for every
spacing-keyed walk threshold (the full-width scan was pitch 26 px / 34 windings).

Per slice: per-slice Otsu film mask (`otsu` from diag_zip_zsurvey -- see the doc's
gotcha about hand-rolled variants); roll centre = centroid of the hole-filled
largest film component (eccentric spiral => good to ~a pitch, enough here);
720 radial rays sampled at 1 px; per ray the film RUNS (>= --min-run px) give the
run count, run thickness and start-to-start spacing. Touching windings merge two
runs into one, so spacing is reported as the MEDIAN and count as p50/p90.

Writes geometry.json, per-slice thumbnails (centre + film extent drawn) and native
zoom crops at four azimuths, so the numbers can be checked by eye.

  python Scripts/diag_hr_geometry.py --dir 01_Mickey_hr/h5 --z 600,1300,2000 --out-dir D
"""
import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import tifffile as tf
from scipy.ndimage import binary_fill_holes, label, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_zip_zsurvey import otsu  # noqa: E402


def read_slice(d, z):
    pat = re.compile(r"_(\d+)\.(h5|tif)$", re.I)
    for n in os.listdir(d):
        m = pat.search(n)
        if m and int(m.group(1)) == z:
            p = os.path.join(d, n)
            if n.endswith(".h5"):
                with h5py.File(p, "r") as f:
                    return f["image"][...], p
            return tf.imread(p), p
    raise SystemExit(f"no slice z{z} in {d}")


def runs_1d(v, min_run):
    """(start, length) of True runs >= min_run."""
    v = np.concatenate([[False], v, [False]]).astype(np.int8)
    d = np.diff(v)
    s = np.nonzero(d == 1)[0]; e = np.nonzero(d == -1)[0]
    L = e - s
    k = L >= min_run
    return s[k], L[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--z", required=True, help="comma list of slice indices")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-rays", type=int, default=720)
    ap.add_argument("--min-run", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    from PIL import Image, ImageDraw

    out = {}
    for z in [int(s) for s in args.z.split(",")]:
        a, path = read_slice(args.dir, z)
        dt0 = str(a.dtype)
        a = a.astype(np.float32)
        sub = a[::4, ::4]
        thr = otsu(sub)
        m = a > thr
        # roll = largest component of the coarse mask, holes filled
        ms = m[::4, ::4]
        lab, n = label(ms)
        if n == 0:
            out[z] = {"error": "no film"}; continue
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        disk = binary_fill_holes(lab == big)
        yy, xx = np.nonzero(disk)
        cy, cx = 4 * yy.mean(), 4 * xx.mean()
        r_disk = 4 * np.sqrt(disk.sum() / np.pi)
        # rays
        R = int(min(r_disk * 1.15, min(a.shape) / 2 - 2))
        rr = np.arange(0, R, 1.0)
        th = np.linspace(0, 2 * np.pi, args.n_rays, endpoint=False)
        cnt, spac, thick, rlo, rhi, ac_pitch = [], [], [], [], [], []
        for t in th:
            ys = cy + rr * np.sin(t); xs = cx + rr * np.cos(t)
            ok = (ys >= 0) & (ys < a.shape[0] - 1) & (xs >= 0) & (xs < a.shape[1] - 1)
            prof = map_coordinates(a, [ys[ok], xs[ok]], order=1)
            # threshold-free pitch: autocorrelation of the profile over the film
            # annulus (the raw contrast here is edge-dominated: local tomography
            # loses the absorption offset, so base ~ air in grey level)
            seg_ = prof[int(0.6 * r_disk):int(0.98 * r_disk)]
            if len(seg_) > 200:
                s0 = seg_ - seg_.mean()
                ac = np.correlate(s0, s0, "full")[len(s0) - 1:]
                lo_l, hi_l = 12, 60
                ac_pitch.append(lo_l + int(np.argmax(ac[lo_l:hi_l])))
            v = prof > thr
            s, L = runs_1d(v, args.min_run)
            # drop the bore (runs well inside) via the disk's own extent: keep all,
            # the bore is air so it produces no runs except ring artefacts (thin)
            if len(s) < 3:
                continue
            cnt.append(len(s)); thick.extend(L.tolist())
            spac.extend(np.diff(s).tolist()); rlo.append(s[0]); rhi.append(s[-1] + L[-1])
        cnt = np.array(cnt); spac = np.array(spac, float); thick = np.array(thick, float)
        rec = {
            "path": path, "shape": list(a.shape), "dtype": dt0,
            "otsu": float(thr), "intensity_p1_p50_p99": np.percentile(sub, [1, 50, 99]).tolist(),
            "centre_xy": [float(cx), float(cy)], "r_disk": float(r_disk),
            "film_r_lo_p50": float(np.median(rlo)), "film_r_hi_p50": float(np.median(rhi)),
            "runs_per_ray_p10_p50_p90": np.percentile(cnt, [10, 50, 90]).tolist(),
            "run_thick_p10_p50_p90": np.percentile(thick, [10, 50, 90]).tolist(),
            "spacing_p10_p50_p90": np.percentile(spac, [10, 50, 90]).tolist(),
            "spacing_mode": float(np.bincount(spac.astype(int)).argmax()),
            "autocorr_pitch_p10_p50_p90": np.percentile(ac_pitch, [10, 50, 90]).tolist()
            if ac_pitch else None,
            "est_windings": float((np.median(rhi) - np.median(rlo)) / np.median(ac_pitch))
            if ac_pitch else None,
            "n_rays_used": int(len(cnt)),
        }
        out[z] = rec
        print(z, json.dumps(rec), flush=True)

        # thumbnail
        lo, hi = np.percentile(sub, [1, 99.5])
        im = Image.fromarray((np.clip((sub - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        c4 = (cx / 4, cy / 4)
        for rad, col in ((rec["film_r_lo_p50"], (0, 255, 0)), (rec["film_r_hi_p50"], (255, 0, 0))):
            r4 = rad / 4
            d.ellipse([c4[0] - r4, c4[1] - r4, c4[0] + r4, c4[1] + r4], outline=col, width=2)
        d.ellipse([c4[0] - 4, c4[1] - 4, c4[0] + 4, c4[1] + 4], fill=(255, 255, 0))
        im.save(os.path.join(args.out_dir, f"thumb_z{z:04d}.png"))
        # native crops at 4 azimuths, mid-film radius
        rm = 0.5 * (rec["film_r_lo_p50"] + rec["film_r_hi_p50"])
        tiles = []
        for t in (0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
            y0 = int(cy + rm * np.sin(t)); x0 = int(cx + rm * np.cos(t))
            c = a[max(0, y0 - 200):y0 + 200, max(0, x0 - 200):x0 + 200]
            lo, hi = np.percentile(c, [1, 99.5])
            tiles.append(Image.fromarray((np.clip((c - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)).resize((400, 400)))
        M = Image.new("L", (1610, 400), 255)
        for i, t in enumerate(tiles):
            M.paste(t, (i * 403, 0))
        M.save(os.path.join(args.out_dir, f"crops_z{z:04d}.png"))
    json.dump(out, open(os.path.join(args.out_dir, "geometry.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
