"""Independent proof that the restitch render read the RESTITCHED CT values.

Inputs: two `v13 --dump-rows` dumps on the SAME rows -- one on the restitched volume
(with --ct-remap), one on the original volume (no remap) -- plus both full strips and,
optionally, TIFs re-fetched straight from RA.

  A. Source files: each dumped row names the h5 it read; assert it lives in --new-dir.
  B. h5 == RA original: re-fetched TIFs are pixel-identical to our converted h5.
  C. Volumes differ: restitched slice vs the original slice at the matched z.
  D. Re-sampling from scratch: read the named h5 with h5py directly, normalise, and
     resample at the dumped positions with scipy map_coordinates (NOT the pipeline's
     bilinear_sample) incl. the transverse taps -> must equal the dumped row; the
     SAME positions on the ORIGINAL volume must NOT.
  E. The dumped rows sit verbatim inside the delivered strips (located per winding,
     since --seam-bridge inserts blended columns between windings).
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


def load_h5(p):
    with h5py.File(p, "r") as f:
        return np.asarray(f["image"], dtype=np.float32)


def norm(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def slice_map(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def resample(img, xs, ys, Ks, tw):
    """Transverse-averaged order-1 resampling, normal computed per winding."""
    offs = np.linspace(-(tw - 1) / 2.0, (tw - 1) / 2.0, tw)
    out, c = [], 0
    H, W = img.shape
    for K in Ks:
        x, y = xs[c:c + K].astype(np.float64), ys[c:c + K].astype(np.float64)
        if tw > 1:
            dx, dy = np.gradient(x), np.gradient(y)
            L = np.hypot(dx, dy) + 1e-9
            nx, ny = -dy / L, dx / L
        else:
            nx = ny = np.zeros_like(x)
        acc = np.zeros(K)
        for t in offs:
            xx, yy = x + t * nx, y + t * ny
            v = map_coordinates(img, [yy, xx], order=1, mode="constant", cval=0.0)
            v[(xx < 0) | (xx > W - 1) | (yy < 0) | (yy > H - 1)] = 0.0
            acc += v
        out.append(acc / tw); c += K
    return np.concatenate(out)


def locate_in_strip(raw, srow, Ks, max_extra):
    """Find each winding's raw segment in the strip row; return max |diff| over all."""
    worst, c, extra = 0.0, 0, 0
    for K in Ks:
        seg = raw[c:c + K]; best = None
        for e in range(extra, max_extra + 1):
            s = srow[c + e:c + e + K]
            if len(s) < K:
                break
            d = float(np.max(np.abs(s - seg)))
            if best is None or d < best[0]:
                best = (d, e)
            if d == 0.0:
                break
        worst = max(worst, best[0]); extra = best[1]; c += K
    return worst, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-new", required=True); ap.add_argument("--dump-old", required=True)
    ap.add_argument("--strip-new", required=True); ap.add_argument("--strip-old", required=True)
    ap.add_argument("--new-dir", required=True); ap.add_argument("--old-dir", required=True)
    ap.add_argument("--tif-dir", default=None)
    a = ap.parse_args()
    N, O = np.load(a.dump_new), np.load(a.dump_old)
    Ks, tw = N["K"], int(N["transverse_px"])
    assert np.array_equal(N["z"], O["z"]) and np.array_equal(Ks, O["K"])
    newdir = os.path.realpath(a.new_dir); old_map = slice_map(a.old_dir)
    SN, SO = np.load(a.strip_new, mmap_mode="r"), np.load(a.strip_old, mmap_mode="r")

    print("== A. source file of every dumped row ==")
    for z, s, so in zip(N["z"], N["src"], O["src"]):
        ok = os.path.realpath(str(s)).startswith(newdir + os.sep)
        print(f"  row z{z}: restitch render read {s}  [{'IN new dir' if ok else 'NOT IN NEW DIR'}]"
              f" | original render read {so}")
        assert ok

    if a.tif_dir:
        print("== B. our h5 vs TIFs re-fetched from RA ==")
        import tifffile
        nmap = slice_map(a.new_dir)
        for p in sorted(glob.glob(os.path.join(a.tif_dir, "*.tif"))):
            z = int(re.search(r"(\d+)\.tif$", p).group(1))
            t, h = tifffile.imread(p), None
            with h5py.File(nmap[z], "r") as f:
                h = f["image"][...]
            print(f"  {os.path.basename(p)} {t.shape} {t.dtype} vs {os.path.basename(nmap[z])}: "
                  f"{'IDENTICAL' if (t.shape == h.shape and np.array_equal(t, h)) else 'DIFFERENT'}")

    print("== C/D/E per row ==")
    for i, z in enumerate(N["z"]):
        src = str(N["src"][i]); tx, ty = float(N["tx"][i]), float(N["ty"][i])
        img_new = load_h5(src); img_old = load_h5(old_map[int(z)])
        # C: raw volumes, old shifted by t onto the new grid (integer part only, coarse)
        sx, sy = int(round(tx)), int(round(ty))
        Hc = min(img_new.shape[0], img_old.shape[0]) - 8; Wc = min(img_new.shape[1], img_old.shape[1]) - 8
        A_ = img_new[4 + sy:Hc + sy, 4 + sx:Wc + sx]; B_ = img_old[4:Hc, 4:Wc]
        same = np.array_equal(img_new, img_old) if img_new.shape == img_old.shape else False
        cc = np.corrcoef(A_[::4, ::4].ravel(), B_[::4, ::4].ravel())[0, 1]
        print(f"  z{z}: restitched {os.path.basename(src)} {img_new.shape} vs original "
              f"{os.path.basename(old_map[int(z)])} {img_old.shape}: identical={same}, "
              f"corr {cc:.4f}, mean|raw diff| {np.mean(np.abs(A_ - B_)):.1f} (raw units)")
        # D: independent resampling
        rN = resample(norm(img_new), N["xs"][i], N["ys"][i], Ks, tw)
        rO_same_pos = resample(norm(img_old), N["xs"][i], N["ys"][i], Ks, tw)
        rO_own = resample(norm(img_old), O["xs"][i], O["ys"][i], Ks, tw)
        dN = np.abs(rN - N["rows"][i]); dX = np.abs(rO_same_pos - N["rows"][i])
        dO = np.abs(rO_own - O["rows"][i])
        print(f"      restitch row vs from-scratch RESTITCHED sampling: max {dN.max():.2e}  "
              f"median {np.median(dN):.2e}")
        print(f"      restitch row vs ORIGINAL volume at the same positions: max {dX.max():.3f}  "
              f"median {np.median(dX):.4f}")
        print(f"      original row vs from-scratch ORIGINAL sampling: max {dO.max():.2e}")
        print(f"      positions: restitch - original = mean ({np.mean(N['xs'][i] - O['xs'][i]):+.3f},"
              f"{np.mean(N['ys'][i] - O['ys'][i]):+.3f}) px  (= t {tx:+.3f},{ty:+.3f})")
        # E: inside the delivered strips
        r = int(N["row_index"][i]); extra = SN.shape[1] - int(Ks.sum())
        wN, eN = locate_in_strip(N["rows"][i], np.asarray(SN[r]), Ks, extra)
        wO, eO = locate_in_strip(O["rows"][i], np.asarray(SO[r]), Ks, extra)
        print(f"      delivered RESTITCH strip row {r}: max |diff| {wN:.2e} (bridge cols {eN}/{extra}); "
              f"delivered ORIGINAL strip row {r}: max |diff| {wO:.2e}")


if __name__ == "__main__":
    main()
