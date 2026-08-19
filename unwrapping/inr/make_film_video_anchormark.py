"""Film video with each z-slice marked WALKED (red) or INTERPOLATED (blue).

The unroll only walks the emulsion on z-slices where segmentation exists. Every
other CT slice gets its winding curve by interpolating (x,y) across z between
the nearest walked anchors. With the 25 gapped seg chunks that is 500 walked of
1184 slices -- so ~58% of the film's WIDTH is interpolated geometry, in bands of
~20 measured slices followed by ~27 interpolated ones.

A video frame is `strip[:, cell]`, i.e. strip ROWS (= CT z = film width) run
vertically on screen. So the marking appears as horizontal stripes: red where the
geometry was measured, blue where it was interpolated. Both tints are light so
the picture stays readable -- the point is to see whether artifacts line up with
the interpolated bands.

Cell cutting is imported from make_film_video so the framing is identical to the
normal render.

Usage:
  python -m unwrapping.inr.make_film_video_anchormark \
      <strip.npy> <out.mp4> --z-index <z_index.npy> --geom <matched_walks.npz> \
      [--phase-lock global] [--reverse] [--invert] [--height 720] [--alpha 0.16]
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from unwrapping.inr.make_film_video import (
    estimate_pitch, refine_pitch, refine_pitch_phase, periodic_component,
    detect_boundaries, tracked_bounds,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy"); ap.add_argument("out")
    ap.add_argument("duration", nargs="?", type=float, default=9.0)
    ap.add_argument("--z-index", required=True,
                    help="z_index.npy written by the render = the CT z of each "
                         "strip row.")
    ap.add_argument("--geom", required=True,
                    help="matched_walks.npz — its z_anchor lists the WALKED z.")
    ap.add_argument("--alpha", type=float, default=0.16,
                    help="Tint strength (0..1). 0.16 is visible without hiding "
                         "the picture.")
    ap.add_argument("--pitch", type=float, default=0)
    ap.add_argument("--pitch-method", choices=["phase", "autocorr"], default="phase")
    ap.add_argument("--phase-lock", choices=["adaptive", "global", "track", "none"],
                    default="global")
    ap.add_argument("--invert-phase", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--rot90", type=int, default=0,
                    help="Rotate frames by N x 90 deg COUNTER-clockwise "
                         "(1 = CCW quarter turn).")
    ap.add_argument("--side-by-side", action="store_true",
                    help="Emit UNMARKED (left) | MARKED (right) in one frame.")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="Omit the captions above the side-by-side panels.")
    args = ap.parse_args()

    s = np.load(args.npy, mmap_mode="r")
    Z, W = s.shape
    z_row = np.load(args.z_index).astype(int)                 # (Z,) CT z per row
    if len(z_row) != Z:
        raise SystemExit(f"z-index has {len(z_row)} entries but strip has {Z} rows")
    g = np.load(args.geom, allow_pickle=True)
    z_walked = set(int(round(v)) for v in np.asarray(g["z_anchor"], float))
    walked = np.array([z in z_walked for z in z_row], bool)
    print(f"strip {s.shape}; {walked.sum()}/{Z} rows WALKED "
          f"({100*walked.mean():.1f}%), {(~walked).sum()} interpolated")
    # describe the banding so the stripes in the video are interpretable
    edges = np.flatnonzero(np.diff(walked.astype(int)) != 0) + 1
    runs = np.diff(np.concatenate([[0], edges, [Z]]))
    lab = walked[np.concatenate([[0], edges])]
    wr = runs[lab]; ir = runs[~lab]
    if wr.size and ir.size:
        print(f"  walked runs: {wr.size} of median {int(np.median(wr))} slices; "
              f"interpolated runs: {ir.size} of median {int(np.median(ir))} slices")

    col = np.asarray(s).mean(axis=0)
    pitch_c = int(round(args.pitch)) if args.pitch > 0 else estimate_pitch(col)
    if args.pitch > 0:
        pitch_f = float(args.pitch)
    elif args.pitch_method == "phase":
        pitch_f = refine_pitch_phase(col, pitch_c)
    else:
        pitch_f = refine_pitch(col, pitch_c)

    if args.phase_lock == "none":
        n0 = int(W / pitch_f)
        bounds = np.round(np.arange(n0 + 1) * pitch_f).astype(int)
    elif args.phase_lock == "track":
        bounds = np.round(tracked_bounds(col, pitch_c, pitch_f,
                                         args.invert_phase)).astype(int)
    elif args.phase_lock == "global":
        p = periodic_component(col, pitch_c)
        sig = -p if args.invert_phase else p
        score = [sig[np.round(np.arange(o, W - pitch_f, pitch_f)).astype(int)].sum()
                 for o in range(pitch_c)]
        o0 = int(np.argmax(score))
        n0 = int((W - o0) / pitch_f)
        bounds = np.round(o0 + np.arange(n0 + 1) * pitch_f).astype(int)
    else:
        bounds = detect_boundaries(col, pitch_c, args.invert_phase)
    bounds = np.asarray(bounds)
    n = len(bounds) - 1
    widths = np.diff(bounds)
    fps = max(6, int(round(n / args.duration)))
    order = range(n - 1, -1, -1) if args.reverse else range(n)
    samp = np.asarray(s[:, ::50], dtype=np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    print(f"pitch~{pitch_f:.2f} -> {n} cells, fps={fps}, dur={n/fps:.1f}s", flush=True)

    # per-row tint colour: red = walked (measured), blue = interpolated
    tint = np.zeros((Z, 3), np.float32)
    tint[walked] = (1.0, 0.15, 0.15)
    tint[~walked] = (0.20, 0.45, 1.0)
    a = float(args.alpha)

    SEP, PAD = 8, 34                       # separator px, caption band px
    font = None
    if args.labels and args.side_by_side:
        try:                               # matplotlib always ships DejaVuSans
            from matplotlib import font_manager
            from PIL import ImageFont
            font = ImageFont.truetype(
                font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")), 22)
        except Exception:
            font = None

    def compose(fr):
        """grayscale cell -> the RGB frame to write (rotate / side-by-side)."""
        gray = np.repeat(fr[..., None], 3, axis=2)
        mark = fr[..., None] * (1.0 - a) + tint[:, None, :] * a
        if args.rot90:
            gray = np.rot90(gray, args.rot90)      # numpy rot90 is CCW
            mark = np.rot90(mark, args.rot90)
        if not args.side_by_side:
            return np.clip(mark, 0, 1)
        h = gray.shape[0]
        sep = np.ones((h, SEP, 3), np.float32)
        return np.clip(np.concatenate([gray, sep, mark], axis=1), 0, 1)

    probe = compose(np.zeros((Z, int(np.median(widths))), np.float32))
    ph, pw = probe.shape[:2]
    H = args.height
    Wf = int(round(H * pw / ph)); Wf += Wf % 2
    head = PAD if font is not None else 0
    w = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8,
                           macro_block_size=1)
    for c, i in enumerate(order):
        fr = np.asarray(s[:, bounds[i]:bounds[i + 1]], dtype=np.float32)
        fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        rgb = compose(fr)
        im = Image.fromarray((rgb * 255).astype(np.uint8)).resize((Wf, H))
        if font is not None:
            canvas = Image.new("RGB", (Wf, H + head), "white")
            canvas.paste(im, (0, head))
            dr = ImageDraw.Draw(canvas)
            half = Wf // 2
            dr.text((10, 7), "unmarked", fill=(0, 0, 0), font=font)
            dr.text((half + 10, 7), "red = walked   blue = interpolated",
                    fill=(0, 0, 0), font=font)
            im = canvas
        w.append_data(np.asarray(im))
        if c % 50 == 0:
            print(f"  frame {c}/{n}", flush=True)
    w.close()

    # a still, at full row resolution, so the banding can be inspected closely
    i = list(order)[len(list(order)) // 2]
    fr = np.asarray(s[:, bounds[i]:bounds[i + 1]], dtype=np.float32)
    fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
    if args.invert:
        fr = 1.0 - fr
    Image.fromarray((compose(fr) * 255).astype(np.uint8)).save(
        args.out + ".marked_cell.png")
    # and the whole strip, downscaled along arc only (rows kept 1:1)
    stepc = max(1, W // 3000)
    sm = np.clip((np.asarray(s[:, ::stepc], np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    if args.invert:
        sm = 1.0 - sm
    rgb = np.clip(sm[..., None] * (1.0 - a) + tint[:, None, :] * a, 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(
        args.out + ".marked_strip.png")     # strip still stays unrotated
    print(f"Done -> {args.out} ({H}x{Wf}); stills {args.out}.marked_cell.png "
          f"+ .marked_strip.png")


if __name__ == "__main__":
    main()
