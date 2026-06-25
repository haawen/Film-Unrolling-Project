"""Diagnose the film frame pitch: print top autocorrelation lags + frame montages."""
import sys
import numpy as np
from PIL import Image

npy, out_dir = sys.argv[1], sys.argv[2]
s = np.load(npy, mmap_mode="r"); Z, W = s.shape
col = np.asarray(s).mean(axis=0)
k = 2001; pad = np.pad(col, k // 2, mode="edge")
sm = np.convolve(pad, np.ones(k) / k, "valid")[:len(col)]
d = col - sm; d -= d.mean()
ac = np.correlate(d, d, "full")[len(d) - 1:]
# top local-max lags in [200, 3000]
lo, hi = 200, min(3000, len(ac) - 2)
peaks = [(l, ac[l]) for l in range(lo, hi) if ac[l] > ac[l - 1] and ac[l] > ac[l + 1]]
peaks.sort(key=lambda t: -t[1])
print("top autocorr lags (px):", [(int(l), round(float(v), 1)) for l, v in peaks[:8]])

lo_, hi_ = np.percentile(np.asarray(s[:, ::50], dtype=np.float32), [1, 99])
for pitch in sorted({int(sys.argv[3]) if len(sys.argv) > 3 else 1000,
                     peaks[0][0] if peaks else 1000}):
    c0 = W // 3
    tiles = []
    for j in range(5):
        fr = np.asarray(s[:, c0 + j * pitch:c0 + (j + 1) * pitch], dtype=np.float32)
        fr = np.clip((fr - lo_) / (hi_ - lo_ + 1e-6), 0, 1)
        tiles.append(Image.fromarray((fr * 255).astype(np.uint8)).resize((220, 380)))
    mon = Image.new("L", (220 * 5, 380), "white")
    for j, t in enumerate(tiles):
        mon.paste(t, (j * 220, 0))
    mon.save(f"{out_dir}/pitch_{pitch}_frames.png")
    print(f"saved pitch_{pitch}_frames.png")
print("done")
