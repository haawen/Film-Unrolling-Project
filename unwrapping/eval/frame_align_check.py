"""Is a rendered film framed on the same piece of film as the GT scan?

Compares the TEMPORAL MEDIAN of two videos, not individual frames.  This reel
is largely one static scene, so the median is dominated by the fixed background
(wall, fence, chair) while the animated characters average away -- a stable
template that exists in both videos.  Frame-by-frame comparison does not work
here: consecutive GT frames are near-identical, so the best-matching GT frame
for one of ours is ambiguous over a +-6 frame window at correlations of ~0.1.

The search is a scale sweep with an FFT cross-correlation over all translations
at each scale.  A plain gradient-descent registration (findTransformECC) stalls
at cc = 0.17 on this pair and returns parameters that look plausible and are
wrong; the exhaustive sweep finds cc = 0.60.  When two registrations disagree,
the one that reports the higher correlation on the same features is the one to
believe -- and a low absolute correlation is itself the warning.

Reports the answer as framing error: how much more film our window spans, and
how far its centre sits from GT's, in fractions of the frame.

    python -m unwrapping.eval.frame_align_check --a ours.mp4 \
        --b unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4 \
        --a-margin 0.04 --out-dir <dir>
"""

import argparse
import os

import numpy as np


def load_median(path, margin=0.0):
    """Temporal median of a video, optionally with a rendered margin cropped
    off so the frame holds the picture and nothing else."""
    import cv2
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255)
    cap.release()
    if not fr:
        raise SystemExit(f"no frames in {path}")
    F = np.stack(fr)
    if margin > 0:
        m = margin / (1 + 2 * margin)
        h, w = F.shape[1:]
        F = F[:, int(m * h):int((1 - m) * h), int(m * w):int((1 - m) * w)]
    return np.median(F, axis=0), len(fr)


def _feat(x, ds, size=None):
    import cv2
    if size is not None:
        x = cv2.resize(x, size)
    x = cv2.resize(x, (x.shape[1] // ds, x.shape[0] // ds))
    x = cv2.GaussianBlur(x, (0, 0), 1.5)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
    m = np.hypot(gx, gy)
    return (m - m.mean()) / (m.std() + 1e-9)


def align(med_a, med_b, ds=2, s_lo=0.85, s_hi=1.25, s_step=0.005):
    """Best (cc, sx, sy, tx, ty) taking A onto B, translations in B pixels."""
    import cv2
    H0, W0 = med_b.shape
    A = _feat(med_a, ds, size=(W0, H0))
    B = _feat(med_b, ds)
    H, W = B.shape
    Bf = np.fft.rfft2(B)
    nB = np.linalg.norm(B)
    best = (-9.0, 1.0, 1.0, 0, 0)
    for sx in np.arange(s_lo, s_hi + 1e-9, s_step):
        for sy in np.arange(s_lo, s_hi + 1e-9, s_step):
            M = np.array([[sx, 0, W / 2 - sx * W / 2],
                          [0, sy, H / 2 - sy * H / 2]], np.float32)
            Aw = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderValue=0)
            cc = np.fft.irfft2(np.conj(np.fft.rfft2(Aw)) * Bf, s=(H, W))
            cc /= (np.linalg.norm(Aw) * nB + 1e-9)
            k = np.unravel_index(np.argmax(cc), cc.shape)
            if cc[k] > best[0]:
                ty = k[0] - H if k[0] > H // 2 else k[0]
                tx = k[1] - W if k[1] > W // 2 else k[1]
                best = (float(cc[k]), float(sx), float(sy), int(tx) * ds,
                        int(ty) * ds)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="our render")
    ap.add_argument("--b", required=True, help="GT picture video")
    ap.add_argument("--a-margin", type=float, default=0.0,
                    help="Margin the render carries outside the picture, so it "
                         "can be cropped off before comparing.")
    ap.add_argument("--b-margin", type=float, default=0.0)
    ap.add_argument("--ds", type=int, default=2)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="align")
    args = ap.parse_args()

    ma, na = load_median(args.a, args.a_margin)
    mb, nb = load_median(args.b, args.b_margin)
    print(f"A {args.a}  {na} frames, median {ma.shape}", flush=True)
    print(f"B {args.b}  {nb} frames, median {mb.shape}", flush=True)

    cc, sx, sy, tx, ty = align(ma, mb, args.ds)
    H, W = mb.shape
    print(f"\ncc={cc:.3f}  sx={sx:.3f} sy={sy:.3f}  tx={tx:+d} ty={ty:+d} px",
          flush=True)
    print("framing of A relative to B:", flush=True)
    print(f"  across: spans {(1 / sx - 1) * 100:+.1f} % more film, "
          f"centre off {-tx / W * 100:+.1f} % of the width", flush=True)
    print(f"  along : spans {(1 / sy - 1) * 100:+.1f} % more film, "
          f"centre off {-ty / H * 100:+.1f} % of the height", flush=True)
    print("\nto correct, pass to border_render:", flush=True)
    print(f"  --along-shift {ty / H:+.3f}  --along-scale {sy:.3f}", flush=True)
    print(f"  --across-shift {tx / W:+.3f} --across-scale {sx:.3f}", flush=True)
    if cc < 0.35:
        print("\nWARNING: correlation is low; treat these numbers as unproven.",
              flush=True)

    if args.out_dir:
        import cv2
        os.makedirs(args.out_dir, exist_ok=True)
        def nrm(x):
            lo, hi = np.percentile(x, [2, 98])
            return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
        a = cv2.resize(nrm(ma), (W, H))
        sheet = (np.concatenate([a, nrm(mb)], axis=1) * 255).astype(np.uint8)
        p = os.path.join(args.out_dir, f"{args.tag}_medians.png")
        cv2.imwrite(p, sheet)
        with open(os.path.join(args.out_dir, f"{args.tag}.txt"), "w") as f:
            f.write(f"cc {cc:.4f} sx {sx:.4f} sy {sy:.4f} tx {tx} ty {ty}\n")
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
