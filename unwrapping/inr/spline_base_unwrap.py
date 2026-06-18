"""
Standalone spline-base unwrapper (polar r(phi) about a fixed center).

New analytical base for unwrapping that replaces the Archimedean spiral fit.
Instead of assuming a perfect r(theta) = r_inner + a*theta spiral with a global
spool center, we FIT a smooth function to the segmentation directly:

  1. Detect spool center + per-winding emulsion centerlines (raycast detector).
  2. Build a point cloud (phi, r) where, for winding k and ray angle theta,
         phi = 2*pi*k + ((theta - theta_seam) mod 2*pi)      (cumulative angle)
         r   = per_angle[k, theta]                            (emulsion radius)
     This is continuous across winding seams (winding k's end meets k+1's start).
  3. Fit a smoothing spline r(phi) (scipy UnivariateSpline) — denoises the
     per-winding jitter while following genuine non-uniform spacing.
  4. Parameterize columns uniformly in ARC LENGTH (matching the synthetic GT
     u_map convention) by inverting s(phi) = integral sqrt(r^2 + r'^2) dphi.
  5. Render the strip by sampling CT intensity along the curve (multi-tap
     through the emulsion), stacked over z. No INR, no training.

This is a standalone unwrapper: render -> evaluate. On synthetic it reports
row_corr / SSIM vs GT; on real Mickey it reports strip_score proxies.

Usage:
  python -m unwrapping.inr.spline_base_unwrap \
      --data-dir <dir> --out-dir <out> --max-slices 48 \
      --smooth-per-pt 4.0 --multitap-n 5 --multitap-delta-px 2.5 \
      [--gt-npz ground_truth.npz]
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.surface_data import detect_winding_boundaries_raycast
from unwrapping.inr.sanity_winding_strip import load_stack, bilinear_sample


def build_spiral_cloud(per_angle, theta_seam, n_rays, seam_exclude_deg=15.0):
    """Build (phi, r) point cloud from per-winding emulsion centerlines.

    per_angle: (n_layers, n_rays) emulsion-centerline radius, ray i -> angle
               2*pi*i/n_rays (absolute, from +x axis).
    Returns sorted (phi, r) with phi = 2*pi*k + wrap(theta - theta_seam).

    seam_exclude_deg: drop cloud points within this angular band of the seam.
        At the seam the windings are radially discontinuous (the spiral steps
        out by ~one layer spacing across the seam ray), so the raycast
        per-winding emulsion-run assignment is unreliable there and produces
        a spike. Excluding the band lets the spline interpolate smoothly across
        the seam instead of chasing the artifact.
    """
    n_layers = per_angle.shape[0]
    ray_theta = np.arange(n_rays) * (2.0 * math.pi / n_rays)       # (n_rays,)
    wrap = np.mod(ray_theta - theta_seam, 2.0 * math.pi)           # (n_rays,)
    band = math.radians(seam_exclude_deg)
    keep = (wrap > band) & (wrap < 2.0 * math.pi - band)
    phi_list, r_list = [], []
    for k in range(n_layers):
        phi_list.append(2.0 * math.pi * k + wrap[keep])
        r_list.append(per_angle[k][keep])
    phi = np.concatenate(phi_list)
    r = np.concatenate(r_list)
    order = np.argsort(phi)
    return phi[order], r[order]


def parse_xy(s):
    a, b = s.split(",")
    return float(a), float(b)


def snap_to_emulsion(px, py, seg_2d):
    """Snap a marked (x, y) to the nearest emulsion (class 2) pixel."""
    ey, ex = np.where(seg_2d == 2)
    i = int(np.argmin((ex - px) ** 2 + (ey - py) ** 2))
    return float(ex[i]), float(ey[i])


def xy_to_phi(px, py, cx, cy, theta_seam, spl, n_layers):
    """Convert a marked point to its spiral position phi = 2*pi*k + wrap(theta).

    The azimuth fixes phi mod 2*pi; the radius picks which winding k by matching
    spl(phi) to the point's radius. Returns the phi whose spline radius is
    closest to the marked radius.
    """
    r = math.hypot(px - cx, py - cy)
    theta = math.atan2(py - cy, px - cx)
    wrap = (theta - theta_seam) % (2.0 * math.pi)
    cand = wrap + 2.0 * math.pi * np.arange(n_layers)
    return float(cand[int(np.argmin(np.abs(spl(cand) - r)))])


class _ArchBase:
    """Archimedean r(phi) = c0 + c1*phi least-squares line, spline-compatible API."""
    def __init__(self, phi, r):
        A = np.vstack([np.ones_like(phi), phi]).T
        (self.c0, self.c1), *_ = np.linalg.lstsq(A, r, rcond=None)

    def __call__(self, phi):
        return self.c0 + self.c1 * np.asarray(phi)

    def derivative(self):
        c1 = self.c1
        return lambda phi: np.full_like(np.asarray(phi, dtype=float), c1)


def fit_radius_spline(phi, r, smooth_per_pt=4.0, k=3):
    """Smoothing spline r(phi). s = smooth_per_pt^2 * N (UnivariateSpline scale).

    UnivariateSpline minimizes sum((r - spline)^2) <= s, so s scales with
    N * sigma^2. smooth_per_pt is the target per-point RMS deviation in px.
    """
    # UnivariateSpline needs strictly increasing x; average duplicate phi.
    phi_u, inv = np.unique(phi, return_inverse=True)
    if len(phi_u) < len(phi):
        r_u = np.zeros_like(phi_u)
        cnt = np.zeros_like(phi_u)
        np.add.at(r_u, inv, r)
        np.add.at(cnt, inv, 1.0)
        r_u = r_u / np.maximum(cnt, 1.0)
    else:
        r_u, phi_u = r, phi_u
    s = (smooth_per_pt ** 2) * len(phi_u)
    spl = UnivariateSpline(phi_u, r_u, k=k, s=s)
    return spl


def arclength_table(spl, phi_max, n_dense=40000, phi_min=0.0):
    """Dense (phi, cumulative arc length s) table along r(phi) in [phi_min, phi_max]."""
    phi_dense = np.linspace(phi_min, phi_max, n_dense)
    r_dense = spl(phi_dense)
    dr_dense = spl.derivative()(phi_dense)
    ds = np.sqrt(r_dense ** 2 + dr_dense ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phi_dense))])
    return phi_dense, s, float(s[-1])


def cols_from_arc(spl, phi_dense, s, col_s):
    """Map arc-length positions col_s -> (col_phi, col_r)."""
    col_phi = np.interp(col_s, s, phi_dense)
    return col_phi, spl(col_phi)


def arclength_columns(spl, phi_max, n_cols, n_dense=40000, phi_min=0.0):
    """Return (col_phi, col_r, s_total) sampled uniformly in arc length."""
    phi_dense, s, s_total = arclength_table(spl, phi_max, n_dense, phi_min=phi_min)
    col_s = np.linspace(0.0, s_total, n_cols)
    col_phi, col_r = cols_from_arc(spl, phi_dense, s, col_s)
    return col_phi, col_r, s_total


def render_strip(image_stack, center, col_phi, theta_seam,
                 spl, multitap_n=5, multitap_delta_px=2.5):
    """Render (Z, n_cols) strip by sampling CT along the spline curve."""
    cy, cx = center
    Z = image_stack.shape[0]
    r_col = spl(col_phi)
    theta = theta_seam + col_phi
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    K = (multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * multitap_delta_px
    strip = np.zeros((Z, col_phi.shape[0]), dtype=np.float32)
    for z in range(Z):
        acc = np.zeros(col_phi.shape[0], dtype=np.float64)
        for off in taps:
            r = r_col + off
            acc += bilinear_sample(image_stack[z], cx + r * cos_t, cy + r * sin_t)
        strip[z] = (acc / len(taps)).astype(np.float32)
    xs = cx + r_col * cos_t
    ys = cy + r_col * sin_t
    return strip, (xs, ys)


def save_fit_diagnostic(phi, r, spl, phi_max, out_dir, n_layers):
    """Plot the (phi, r) cloud vs the fitted spline + analytical Archimedean."""
    phi_g = np.linspace(0, phi_max, 4000)
    # Analytical Archimedean reference: r_inner + a*phi fit by least squares.
    A = np.vstack([np.ones_like(phi), phi]).T
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    r_arch = coef[0] + coef[1] * phi_g
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.scatter(phi / (2 * math.pi), r, s=1, alpha=0.15, color="gray",
               label="raycast centerlines (cloud)")
    ax.plot(phi_g / (2 * math.pi), spl(phi_g), "r-", lw=1.5,
            label="smoothing spline r(phi)")
    ax.plot(phi_g / (2 * math.pi), r_arch, "b--", lw=1.0,
            label="analytical Archimedean (lstsq)")
    ax.set_xlabel("winding number (phi / 2pi)")
    ax.set_ylabel("radius (px)")
    ax.set_title(f"Radius vs cumulative angle — {n_layers} windings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radius_fit.png"), dpi=150)
    plt.close()
    # Residual of cloud about spline (denoising quality).
    resid = r - spl(phi)
    return float(np.sqrt(np.mean(resid ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--n-rays", type=int, default=1440)
    ap.add_argument("--smooth-per-pt", type=float, default=1.5,
                    help="Target per-point RMS deviation (px) of the spline fit.")
    ap.add_argument("--seam-exclude-deg", type=float, default=15.0,
                    help="Drop cloud points within this band of the seam ray.")
    ap.add_argument("--base", choices=["spline", "archimedean"], default="spline",
                    help="Base curve: smoothing spline (new) or Archimedean "
                         "least-squares line through the same cloud (baseline).")
    ap.add_argument("--overlay-only", action="store_true",
                    help="Only (re)draw curve_overlay.png — skip strip render, "
                         "GT eval, npz. Fast way to restyle the overlay.")
    ap.add_argument("--overlay-linewidth", type=float, default=1.2)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--pixels-per-winding", type=int, default=1440)
    ap.add_argument("--multitap-n", type=int, default=5)
    ap.add_argument("--multitap-delta-px", type=float, default=2.5)
    ap.add_argument("--gt-npz", type=str, default=None)
    ap.add_argument("--inspect", action="store_true",
                    help="Render sanity-style true-1:1 arc-length windows at a "
                         "few radii (single-tap, true z height, pixel-exact raw) "
                         "instead of the full undersampled strip. For visual "
                         "inspection of content fidelity.")
    ap.add_argument("--n-windows", type=int, default=4,
                    help="Inspect: number of windows along the roll (inner->outer).")
    ap.add_argument("--window-arc-px", type=int, default=2400,
                    help="Inspect: arc length (px) per window, sampled 1:1.")
    ap.add_argument("--start-xy", default=None,
                    help="Manual spiral START as 'x,y' px on the reference slice "
                         "(skips the degenerate inner core). Needs --end-xy.")
    ap.add_argument("--end-xy", default=None,
                    help="Manual spiral END as 'x,y' px (skips the thin outer "
                         "tail). The strip renders over [START, END].")
    ap.add_argument("--no-snap", action="store_true",
                    help="Don't snap the marked start/end to the nearest "
                         "emulsion pixel (default: snap).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading up to {args.max_slices} slices from {args.data_dir} ...")
    image_stack, seg_stack = load_stack(args.data_dir, args.max_slices)
    Z, H, W = image_stack.shape
    print(f"  loaded {Z} slices, {H}x{W}")

    ref = Z // 2
    cy, cx = find_spool_center(seg_stack[ref])
    film = seg_stack[ref] > 0
    ys_f, xs_f = np.where(film)
    r_f = np.sqrt((ys_f - cy) ** 2 + (xs_f - cx) ** 2)
    theta_seam = _detect_seam_angle(film, (cy, cx), r_f.min(), r_f.max())
    print(f"  center=({cy:.1f},{cx:.1f})  seam={math.degrees(theta_seam):.1f} deg")

    _, n_layers, per_angle = detect_winding_boundaries_raycast(
        seg_stack[ref], (cy, cx), n_rays=args.n_rays,
        out_dir=args.out_dir, return_per_angle=True,
    )
    print(f"  raycast: {n_layers} windings")

    phi, r = build_spiral_cloud(per_angle, theta_seam, args.n_rays,
                                seam_exclude_deg=args.seam_exclude_deg)
    if args.base == "archimedean":
        spl = _ArchBase(phi, r)
        print("  base: Archimedean least-squares line")
    else:
        spl = fit_radius_spline(phi, r, smooth_per_pt=args.smooth_per_pt)
    phi_max = float(phi.max())
    fit_rmse = save_fit_diagnostic(phi, r, spl, phi_max, args.out_dir, n_layers)
    print(f"  spline fit RMSE about cloud: {fit_rmse:.2f} px")

    # Manual spiral bounds (marked once per roll; volume is ~z-invariant).
    phi_lo, phi_hi = 0.0, phi_max
    mark_start = mark_end = None
    if args.start_xy and args.end_xy:
        sx, sy = parse_xy(args.start_xy)
        ex, ey = parse_xy(args.end_xy)
        if not args.no_snap:
            sx, sy = snap_to_emulsion(sx, sy, seg_stack[ref])
            ex, ey = snap_to_emulsion(ex, ey, seg_stack[ref])
        phi_lo = xy_to_phi(sx, sy, cx, cy, theta_seam, spl, n_layers)
        phi_hi = xy_to_phi(ex, ey, cx, cy, theta_seam, spl, n_layers)
        if phi_lo > phi_hi:
            phi_lo, phi_hi = phi_hi, phi_lo
            sx, sy, ex, ey = ex, ey, sx, sy
        mark_start, mark_end = (sx, sy), (ex, ey)
        print(f"  manual bounds: start=({sx:.0f},{sy:.0f}) end=({ex:.0f},{ey:.0f}); "
              f"phi=[{phi_lo:.2f},{phi_hi:.2f}] rad "
              f"({(phi_hi - phi_lo) / (2 * math.pi):.2f} turns)")

    # ── inspect mode: true-1:1 arc-length windows at several radii ──
    if args.inspect:
        phi_dense, s_tab, s_total = arclength_table(spl, phi_hi, phi_min=phi_lo)
        print(f"  arc length total = {s_total:.0f} px; rendering "
              f"{args.n_windows} windows × {args.window_arc_px}px @ 1:1")
        Wpx = args.window_arc_px
        # Window centres evenly spaced inner->outer (avoid the very ends/seam).
        centres = np.linspace(0.12, 0.88, args.n_windows) * s_total
        for i, c in enumerate(centres):
            s0 = max(0.0, c - Wpx / 2.0)
            s1 = min(s_total, s0 + Wpx)
            col_s = np.arange(s0, s1)                       # 1 col per arc px
            wphi, _ = cols_from_arc(spl, phi_dense, s_tab, col_s)
            strip, _ = render_strip(
                image_stack, (cy, cx), wphi, theta_seam, spl,
                multitap_n=args.multitap_n,
                multitap_delta_px=args.multitap_delta_px)
            wnd = int(round(phi_dense[np.searchsorted(s_tab, c)] / (2 * math.pi)))
            # Labeled figure, true z height.
            fig, ax = plt.subplots(1, 1, figsize=(min(24, max(6, strip.shape[1] / 60)),
                                                  max(2.0, Z / 12)))
            ax.imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
            ax.set_title(f"Inspect window {i} (~winding {wnd}, arc {int(s0)}–{int(s1)}px, "
                         f"{Z} z-slices, single-tap)")
            ax.set_xlabel("arc length (px)  →"); ax.set_ylabel("z (slice)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"inspect_w{i}.png"), dpi=150)
            plt.close()
            plt.imsave(os.path.join(args.out_dir, f"inspect_w{i}_raw.png"),
                       strip, cmap="gray")
        print(f"Done (inspect). Outputs in {args.out_dir}")
        return

    n_windings = max(1.0, (phi_hi - phi_lo) / (2 * math.pi))
    n_cols = max(2, int(round(n_windings * args.pixels_per_winding)))
    col_phi, col_r, s_total = arclength_columns(spl, phi_hi, n_cols, phi_min=phi_lo)
    print(f"  arc length total = {s_total:.0f} px -> {n_cols} cols "
          f"({n_windings:.2f} windings)")

    # Continuous spiral curve coords (arc-length-ordered → draws as one line).
    theta_col = theta_seam + col_phi
    xs_c = cx + col_r * np.cos(theta_col)
    ys_c = cy + col_r * np.sin(theta_col)

    def draw_overlay():
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.imshow(seg_stack[ref], cmap="gray")
        ax.plot(xs_c, ys_c, "-", color="red",
                linewidth=args.overlay_linewidth, alpha=args.overlay_alpha,
                solid_capstyle="round")
        ax.plot(cx, cy, "c+", markersize=14)
        if mark_start is not None:
            ax.plot(*mark_start, "o", mfc="lime", mec="k", ms=13, label="START")
            ax.plot(*mark_end, "s", mfc="red", mec="k", ms=13, label="END")
            ax.legend(loc="upper right", fontsize=10)
        ax.set_title(f"{args.base} curve overlay on segmentation")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "curve_overlay.png"), dpi=150)
        plt.close()

    if args.overlay_only:
        draw_overlay()
        print(f"Done (overlay-only). Wrote curve_overlay.png in {args.out_dir}")
        return

    strip, _ = render_strip(
        image_stack, (cy, cx), col_phi, theta_seam, spl,
        multitap_n=args.multitap_n, multitap_delta_px=args.multitap_delta_px,
    )

    # Strip figure.
    fig, ax = plt.subplots(1, 1, figsize=(28, max(4, Z / 40)))
    ax.imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Spline-base unwrapped strip ({n_layers} windings, "
                 f"smooth={args.smooth_per_pt}px/pt, no INR)")
    ax.set_xlabel("u (arc length)")
    ax.set_ylabel("z")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "strip_full.png"), dpi=130)
    plt.close()

    # Curve overlay on the reference slice (thin translucent spiral line).
    draw_overlay()

    np.savez_compressed(os.path.join(args.out_dir, "strip.npz"),
                        strip=strip.astype(np.float32), n_layers=n_layers,
                        pixels_per_winding=args.pixels_per_winding)

    # GT-free fit quality: does the base curve land ON the emulsion (class 2)?
    # A well-fit base tracks the emulsion centerline; an over-smoothed / drifted
    # base falls into the air gap or wrong winding -> membership drops.
    emul = (seg_stack == 2).astype(np.float32)
    film = (seg_stack > 0).astype(np.float32)
    em, fm = [], []
    for z in range(Z):
        em.append(bilinear_sample(emul[z], xs_c, ys_c))
        fm.append(bilinear_sample(film[z], xs_c, ys_c))
    em = np.concatenate(em); fm = np.concatenate(fm)
    emul_membership = float(em.mean())
    on_emul_frac = float((em >= 0.5).mean())
    on_film_frac = float((fm >= 0.5).mean())
    print(f"  fit quality: emul_membership={emul_membership:.3f}  "
          f"on_emul_frac={on_emul_frac:.3f}  on_film_frac={on_film_frac:.3f}")

    metrics = {"n_layers": n_layers, "fit_rmse_px": fit_rmse,
               "arc_total_px": s_total,
               "emul_membership": emul_membership,
               "on_emul_frac": on_emul_frac,
               "on_film_frac": on_film_frac}

    if args.gt_npz and os.path.exists(args.gt_npz):
        from unwrapping.inr.surface_eval import evaluate_against_gt
        gt_metrics = evaluate_against_gt(strip, args.gt_npz, n_layers, args.out_dir)
        metrics.update({f"gt_{k}": v for k, v in gt_metrics.items()})
    else:
        # Real: strip_score proxies.
        try:
            from unwrapping.inr.strip_score import score_strip
            sc = score_strip(strip)
            metrics["strip_score"] = sc
            print("strip_score:", json.dumps(sc, indent=2))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  strip_score unavailable: {exc}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
