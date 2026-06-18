"""Unroll ONE winding over the whole z-axis by CONTINUITY-tracking the film band.

Diagnosis (2026-06-16): the raycast per-winding detector keys on the EMULSION
class (2), which segments patchily on the mid-roll chunks, so the per-ray
"k-th emulsion band" count jumps between adjacent rings -> wriggly centerline.
The FILM-BASE band (class 1, actually the whole film = class>=1 sandwich) is
clean and continuous everywhere.

So here we:
  1. Per slice, sample seg along fine rays and find runs of film (class>=1) ->
     each run's center radius is a winding-band center (robust: thick continuous
     band, emulsion sits in its middle so the center IS the emulsion location).
  2. Track ONE physical ring by CONTINUITY: across rays (pick the band-center
     nearest the previous angle's, rejecting big jumps) AND across z (seed each
     anchor from the neighbouring anchor's radius). No winding-index counting,
     so the 34<->33 detected-count drift is irrelevant.
  3. Interpolate the tracked ring (center + radius profile) across z to every
     raw-CT slice and render the close-up. No whole-roll spline.

Anchors (center + per-ray film-band centers) are cached to anchor_filmband_*.npz
(the seg load is the slow ~40-min part; later windings/windows are instant).

Usage:
  python -m unwrapping.inr.unroll_ring_track \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      --winding 17 --angle-start-deg 0 --angle-extent-deg 140 \
      --n-rays 2880 [--use-cached-anchors]
"""

import argparse
import math
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_continuous import (
    load_ct, load_seg_mid, norm_slice, discover_ct_slices,
)


def film_band_centers(seg, center, n_rays, r_min, r_max, min_run=5, max_w=48):
    """Per-ray film-band (class>=1) center radii, padded to max_w with NaN.

    Returns (n_rays, max_w). Each row holds the sorted center radii of the
    contiguous film runs along that ray (emulsion sits mid-band, so the run
    center is the winding centerline).
    """
    cy, cx = center
    ang = np.arange(n_rays) * (2.0 * math.pi / n_rays)
    radii = np.arange(r_min, r_max, 1.0)
    xs = cx + radii[None, :] * np.cos(ang)[:, None]
    ys = cy + radii[None, :] * np.sin(ang)[:, None]
    xi = np.clip(np.round(xs).astype(np.int64), 0, seg.shape[1] - 1)
    yi = np.clip(np.round(ys).astype(np.int64), 0, seg.shape[0] - 1)
    film = seg[yi, xi] >= 1                                     # (n_rays, n_radii)
    FB = np.full((n_rays, max_w), np.nan, dtype=np.float32)
    for i in range(n_rays):
        idx = np.where(film[i])[0]
        if idx.size == 0:
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        cs = [radii[r].mean() for r in runs if r.size >= min_run]
        FB[i, :len(cs)] = cs[:max_w]
    return FB


def build_anchor_cache(seg_dir, n_rays, r_min, r_max, cache_path):
    pairs = discover_volumes(seg_dir)
    print(f"{len(pairs)} segmented chunks, z {pairs[0][2]}-{pairs[-1][3]}")
    z_anchor, cx_a, cy_a, FB_list = [], [], [], []
    for vp, pp, z0, z1 in pairs:
        seg, mid = load_seg_mid(pp)
        cy, cx = find_spool_center(seg)
        FB = film_band_centers(seg, (cy, cx), n_rays, r_min, r_max)
        z_anchor.append(z0 + mid); cx_a.append(cx); cy_a.append(cy); FB_list.append(FB)
        nb = int(np.nanmax(np.sum(~np.isnan(FB), axis=1)))
        print(f"  anchor z{z0 + mid}: center=({cx:.0f},{cy:.0f}) max_bands/ray={nb}",
              flush=True)
    np.savez_compressed(cache_path, z_anchor=np.array(z_anchor, float),
                        cx_a=cx_a, cy_a=cy_a, FB=np.array(FB_list, np.float32),
                        n_rays=n_rays, r_min=r_min, r_max=r_max)
    print(f"  cached -> {cache_path}")


def track_ring(FB, ray_idx, r_seed, max_jump):
    """Follow one ring over ray_idx by nearest-radius continuity (reject jumps)."""
    r = np.empty(len(ray_idx), dtype=np.float64)
    prev = r_seed
    for n, i in enumerate(ray_idx):
        c = FB[i]; c = c[~np.isnan(c)]
        if c.size:
            j = int(np.argmin(np.abs(c - prev)))
            if abs(c[j] - prev) <= max_jump:
                prev = float(c[j])
        r[n] = prev
    return r


def snap_to_emulsion(PA_anchor, ray_idx, r_band, tol, n_rays, smooth_deg=3.0):
    """Shift the smooth band-center track onto the emulsion sublayer.

    For each window ray, find the emulsion centerline (per_angle) nearest the
    band-center radius, but only within +-tol (so it locks onto THIS ring's
    emulsion, not a neighbour). Where no emulsion is within tol (dashed seg),
    leave the offset NaN. Then fill gaps + lightly smooth the offset along the
    arc so the result follows the emulsion without the per-ray detection jitter.
    Returns the emulsion-following radius over ray_idx.
    """
    off = np.full(len(ray_idx), np.nan)
    for n, ri in enumerate(ray_idx):
        cand = PA_anchor[:, ri]; cand = cand[~np.isnan(cand)]
        if cand.size == 0:
            continue
        j = int(np.argmin(np.abs(cand - r_band[n])))
        if abs(cand[j] - r_band[n]) <= tol:
            off[n] = cand[j] - r_band[n]
    if np.all(np.isnan(off)):
        return r_band.copy()
    idx = np.arange(len(off))
    good = ~np.isnan(off)
    off = np.interp(idx, idx[good], off[good])                 # fill gaps
    win = max(1, int(round(smooth_deg / 360.0 * n_rays)))      # moving-avg window
    if win > 1:
        k = np.ones(win) / win
        off = np.convolve(np.pad(off, win, mode="edge"), k, "same")[win:-win]
    return r_band + off


def arclen_resample(ang, r, n_cols=0):
    dr = np.gradient(r, ang)
    ds = np.sqrt(r ** 2 + dr ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(ang))])
    st = float(s[-1])
    if n_cols <= 0:
        n_cols = max(2, int(round(st)))
    cs = np.linspace(0.0, st, n_cols)
    return np.interp(cs, s, ang), np.interp(cs, s, r), st, n_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anchor-cache", default=None)
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--winding", type=int, default=None,
                    help="Seed ring = k-th film band at the reference start ray. "
                         "Default: middle band.")
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=140.0)
    ap.add_argument("--n-rays", type=int, default=2880)
    ap.add_argument("--r-min", type=float, default=400.0)
    ap.add_argument("--r-max", type=float, default=1500.0)
    ap.add_argument("--max-jump-px", type=float, default=11.0,
                    help="Max allowed ring radius change per ray / z-seed (~half "
                         "the layer spacing) — rejects jumps to a neighbour ring.")
    ap.add_argument("--emulsion-offset-px", type=float, default=0.0,
                    help="Constant radial offset from the film-band center "
                         "(applied after any --snap-emulsion).")
    ap.add_argument("--offset-list", default=None,
                    help="Comma list of constant radial offsets (px) to render in "
                         "ONE pass on the smooth band track (e.g. '-8,-4,0,4,8'). "
                         "Positive = outward. Each -> its own strip_off{v}.png. "
                         "Stays smooth (band track shifted by a constant).")
    ap.add_argument("--snap-emulsion", action="store_true",
                    help="(NOT recommended — wriggles) per-ray emulsion snap.")
    ap.add_argument("--emul-cache", default=None,
                    help="anchor_perangle_*.npz path (default: alongside the "
                         "film-band cache).")
    ap.add_argument("--snap-tol-px", type=float, default=8.0)
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--inspect-anchors", action="store_true",
                    help="Instead of rendering the strip, draw the tracked ring "
                         "(+ --emulsion-offset-px) on EACH anchor's CT slice and "
                         "montage them, so broken anchors can be eyeballed.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cache_path = args.anchor_cache or os.path.join(
        os.path.dirname(os.path.normpath(args.out_dir)),
        f"anchor_filmband_r{args.n_rays}.npz")
    if not (args.use_cached_anchors and os.path.exists(cache_path)):
        build_anchor_cache(args.seg_dir, args.n_rays, args.r_min, args.r_max, cache_path)
    d = np.load(cache_path)
    z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
    FB = d["FB"]; n_rays = int(d["n_rays"])
    n_anchor = len(z_anchor)
    print(f"anchors: {n_anchor}")

    # Window ray indices (absolute angle).
    ang_all = np.arange(n_rays) * (2.0 * math.pi / n_rays)
    a0 = math.radians(args.angle_start_deg)
    a1 = a0 + math.radians(args.angle_extent_deg)
    ray_idx = np.where((ang_all >= a0) & (ang_all <= a1))[0]
    wang = ang_all[ray_idx]
    i0 = ray_idx[0]                                            # start ray

    # Seed ring at the reference (middle) anchor.
    ref = n_anchor // 2
    seed_bands = FB[ref, i0]; seed_bands = np.sort(seed_bands[~np.isnan(seed_bands)])
    k = args.winding if args.winding is not None else len(seed_bands) // 2
    k = int(np.clip(k, 0, len(seed_bands) - 1))
    r_seed_ref = float(seed_bands[k])
    print(f"ref anchor z{int(z_anchor[ref])}: {len(seed_bands)} bands, seed "
          f"ring k={k} r0={r_seed_ref:.0f}")

    # Track per anchor, propagating the seed outward in z from the reference.
    r_tracks = [None] * n_anchor
    r_tracks[ref] = track_ring(FB[ref], ray_idx, r_seed_ref, args.max_jump_px)
    for i in range(ref + 1, n_anchor):                        # up in z
        r_tracks[i] = track_ring(FB[i], ray_idx, r_tracks[i - 1][0], args.max_jump_px)
    for i in range(ref - 1, -1, -1):                          # down in z
        r_tracks[i] = track_ring(FB[i], ray_idx, r_tracks[i + 1][0], args.max_jump_px)

    # Snap the smooth band track onto the emulsion sublayer (band track is kept
    # for z-seeding above; snapping only changes where we sample, not ring ID).
    r_render = [r.copy() for r in r_tracks]
    if args.snap_emulsion:
        emul_path = args.emul_cache or os.path.join(
            os.path.dirname(os.path.normpath(args.out_dir)),
            f"anchor_perangle_r{n_rays}.npz")
        PA = np.load(emul_path)["per_angle"]                  # (n_anchor, maxnl, n_rays)
        moved = []
        for i in range(n_anchor):
            r_render[i] = snap_to_emulsion(PA[i], ray_idx, r_tracks[i],
                                           args.snap_tol_px, n_rays)
            moved.append(float(np.mean(r_render[i] - r_tracks[i])))
        print(f"  snapped to emulsion (cache {os.path.basename(emul_path)}); "
              f"mean radial shift {np.mean(moved):+.2f}px")

    z_to_path = discover_ct_slices(args.ct_dir)

    # ── inspect-anchors: draw the tracked ring on each anchor's CT slice ──
    if args.inspect_anchors:
        off = args.emulsion_offset_px
        cw = np.cos(wang); sw = np.sin(wang)
        tiles = []
        from PIL import Image as _Im, ImageDraw as _Dr
        for i in range(n_anchor):
            z = min(z_to_path, key=lambda zz: abs(zz - z_anchor[i]))
            img = norm_slice(load_ct(z_to_path[z]))
            r = r_render[i] + off
            xs = cx_a[i] + r * cw; ys = cy_a[i] + r * sw
            x0 = max(0, int(xs.min()) - 50); x1 = min(img.shape[1], int(xs.max()) + 50)
            y0 = max(0, int(ys.min()) - 50); y1 = min(img.shape[0], int(ys.max()) + 50)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5 * (y1 - y0) / max(1, x1 - x0)))
            ax.imshow(img[y0:y1, x0:x1], cmap="gray")
            ax.plot(xs - x0, ys - y0, "-", color="red", lw=1.2)
            ax.set_title(f"z{z} (anchor {i})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            p = os.path.join(args.out_dir, f"anchor_{i:02d}_z{z}.png")
            plt.tight_layout(); plt.savefig(p, dpi=90); plt.close()
            tiles.append(p)
            print(f"  anchor {i} z{z}: drawn", flush=True)
        # montage 5x5
        ims = [_Im.open(t).convert("RGB").resize((300, 300)) for t in tiles]
        cols = 5; rows = int(np.ceil(len(ims) / cols))
        mon = _Im.new("RGB", (300 * cols, 300 * rows), "white")
        for j, im in enumerate(ims):
            mon.paste(im, ((j % cols) * 300, (j // cols) * 300))
        mon.save(os.path.join(args.out_dir, "anchors_montage.png"))
        print(f"Done (inspect-anchors). {len(tiles)} overlays + montage in {args.out_dir}")
        return

    # Fixed arc-length columns from the reference ring; resample every anchor onto
    # the same angle grid so columns line up across z.
    ang_u, r_u_ref, s_total, n_cols = arclen_resample(wang, r_render[ref])
    Rprof = np.array([np.interp(ang_u, wang, r_render[i]) for i in range(n_anchor)])
    Rprof += args.emulsion_offset_px
    print(f"  arc={s_total:.0f}px -> {n_cols} cols")

    fR = interp1d(z_anchor, Rprof, axis=0, bounds_error=False,
                  fill_value=(Rprof[0], Rprof[-1]))
    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))
    cos_a, sin_a = np.cos(ang_u), np.sin(ang_u)
    K = (args.multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * args.multitap_delta_px

    z_list = sorted(z_to_path)[:: args.z_step]
    print(f"{len(z_to_path)} raw-CT slices; rendering {len(z_list)} (step {args.z_step})")

    # arc overlay on a reference CT slice (smoothness check).
    z_ov = min(z_to_path, key=lambda zz: abs(zz - z_anchor[ref]))
    img_ov = load_ct(z_to_path[z_ov]); rr = fR(z_ov)
    cxo, cyo = float(fcx(z_ov)), float(fcy(z_ov))
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    lo, hi = np.percentile(img_ov, [1, 99])
    ax.imshow(np.clip((img_ov - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
    ax.plot(cxo + rr * cos_a, cyo + rr * sin_a, "-", color="red", lw=2.0)
    ax.plot(cxo, cyo, "c+", ms=14)
    ax.set_title(f"Continuity-tracked ring (seed k={k}) on raw CT z={z_ov}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "arc_overlay.png"), dpi=140)
    plt.close()

    offsets = ([float(v) for v in args.offset_list.split(",")]
               if args.offset_list else [args.emulsion_offset_px])
    strips = {o: np.empty((len(z_list), n_cols), dtype=np.float32) for o in offsets}
    for i, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        r = fR(z); cx, cy = float(fcx(z)), float(fcy(z))
        for o in offsets:
            acc = np.zeros(n_cols, dtype=np.float64)
            for off in taps:
                ro = r + o + off
                acc += bilinear_sample(img, cx + ro * cos_a, cy + ro * sin_a)
            strips[o][i] = (acc / len(taps)).astype(np.float32)
        if i % 100 == 0 or i == len(z_list) - 1:
            print(f"  [{i + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    from PIL import Image as _Im
    for o, strip in strips.items():
        tag = "" if (len(offsets) == 1 and o == 0) else f"_off{o:+.0f}"
        np.save(os.path.join(args.out_dir, f"strip{tag}.npy"), strip)
        lo, hi = np.percentile(strip, [1, 99])
        disp = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
        _Im.fromarray((disp * 255).astype(np.uint8)).save(
            os.path.join(args.out_dir, f"strip{tag}_raw.png"))
        fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(z_list) / 90)))
        ax.imshow(disp, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Ring seed k={k}, radial offset {o:+.0f}px, "
                     f"{len(z_list)} z-slices ({n_cols}px arc)")
        ax.set_xlabel("arc length (px)  →"); ax.set_ylabel("z slice")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"strip{tag}.png"), dpi=130)
        plt.close()
    print(f"Done. {len(offsets)} offset strip(s) {next(iter(strips.values())).shape}. "
          f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
