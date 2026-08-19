"""Direct test on the DELIVERED v12 strip: measure the LOCAL film-cell pitch
along the strip. If the uniform-phi parameterization is the wobble source, the
local pitch must modulate ~+-5% with a period of one winding (~6.7 cells)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

S = np.load("C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v12/wholeroll.npy", mmap_mode="r")
print("strip", S.shape, S.dtype)
# column profile = frame-line signal (dark bars between cells span the full width)
prof = np.asarray(S[::4].mean(axis=0), float)
prof = prof - gaussian_filter1d(prof, 300)          # band-pass around the pitch
prof = gaussian_filter1d(prof, 8)
N = len(prof)
WIN, HOP = 6000, 500                                 # ~6.6 cells per window
lags = np.arange(700, 1150)
cs, pit = [], []
for c0 in range(0, N - WIN, HOP):
    w = prof[c0:c0 + WIN]; w = w - w.mean()
    ac = np.array([np.dot(w[:-L], w[L:]) / (len(w) - L) for L in lags])
    k = int(np.argmax(ac))
    if 0 < k < len(lags) - 1:                        # sub-px parabolic peak
        y0, y1, y2 = ac[k-1], ac[k], ac[k+1]; d = y0 - 2*y1 + y2
        f = 0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0
    else:
        f = 0.0
    cs.append(c0 + WIN / 2); pit.append(lags[k] + f)
cs = np.array(cs); pit = np.array(pit)
ok = (pit > 800) & (pit < 1000)
cs, pit = cs[ok], pit[ok]
ps = gaussian_filter1d(pit, 2)
print(f"local pitch: med {np.median(ps):.1f}  p5 {np.percentile(ps,5):.1f}  "
      f"p95 {np.percentile(ps,95):.1f}  => +-{100*(np.percentile(ps,95)-np.percentile(ps,5))/2/np.median(ps):.1f}%")
# period of the modulation, in columns
m = ps - gaussian_filter1d(ps, 40)
F = np.abs(np.fft.rfft(m - m.mean()))
fr = np.fft.rfftfreq(len(m), d=HOP)
kk = int(np.argmax(F[1:])) + 1
print(f"dominant modulation period = {1/fr[kk]:.0f} strip-columns "
      f"= {1/fr[kk]/np.median(ps):.1f} film cells   "
      f"(one winding ~ 6000px ~ 6.7 cells)")
fig, ax = plt.subplots(2, 1, figsize=(13, 6))
ax[0].plot(cs, ps, lw=0.9); ax[0].axhline(np.median(ps), color="k", lw=0.5)
ax[0].set_title("v12 strip: LOCAL film-cell pitch (px) along the arc")
ax[0].set_xlabel("strip column")
ax[1].plot(cs[:250], ps[:250], lw=1.1); ax[1].axhline(np.median(ps), color="k", lw=0.5)
ax[1].set_title("zoom (first ~125k columns) — periodic = the once-per-turn breathing")
plt.tight_layout(); plt.savefig(r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/30a37fa4-9ac8-4bc3-bd09-b34d845619dc/scratchpad/localpitch.png", dpi=110)
print("saved localpitch.png")
