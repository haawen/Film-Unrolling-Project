"""Side-by-side: ORIGINAL v9 stabilized vs DE-SHADED v9 stabilized (both GT-synced,
233 frames, frame-for-frame aligned). Writes a compare mp4 + a still montage."""
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw

ORIG = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9/film_stabilized_gtsync.mp4"
DESH = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
OUTV = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/compare_original_vs_deshaded.mp4"
OUTM = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/compare_original_vs_deshaded_montage.png"

ra, rb = iio.get_reader(ORIG), iio.get_reader(DESH)
na, nb = ra.count_frames(), rb.count_frames()
N = min(na, nb)
LAB = 34
def labeled(a, b):
    H, W = a.shape[:2]
    canv = np.full((H+LAB, W*2+6, 3), 20, np.uint8)
    canv[LAB:, :W] = a; canv[LAB:, W+6:] = b
    im = Image.fromarray(canv); d = ImageDraw.Draw(im)
    d.text((W//2-40, 8), "ORIGINAL", fill=(255,230,60))
    d.text((W+6+W//2-46, 8), "DE-SHADED", fill=(80,220,255))
    return np.asarray(im)

w = iio.get_writer(OUTV, fps=15, quality=8)
mont_idx = [40, 90, 140, 195]
mont = []
for j in range(N):
    fa = ra.get_data(j); fb = rb.get_data(j)
    fr = labeled(fa, fb)
    w.append_data(fr)
    if j in mont_idx:
        mont.append(fr)
w.close(); ra.close(); rb.close()
Image.fromarray(np.concatenate(mont, axis=0)).save(OUTM)
print(f"wrote {OUTV} ({N} frames) + montage {OUTM}")
