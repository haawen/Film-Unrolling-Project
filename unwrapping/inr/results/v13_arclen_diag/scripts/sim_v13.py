"""LOCAL proof, no cluster time: apply the v13 arc-length re-grid to the EXISTING
v12 strip and measure the film-cell pitch before vs after.

The v12 strip layout is exactly reconstructible:
  block wi = K_wi columns on a uniform phi grid linspace(eps, 2pi-eps, K_wi),
  followed (for wi < last) by M_wi = round(2*eps*K_wi/(2pi-2eps)) seam-bridge
  columns.  eps = 0.3deg. Check: sum(M) must be 338 and the total 211418.

So I can build the same column map v13 would use and resample the delivered
strip's columns with it. The only difference from a true v13 render is that this
resamples already-sampled columns instead of sampling the CT afresh — for a
smooth ~1px-per-column map that is a very close approximation, and it cannot
manufacture the effect we are testing for.

Local pitch is measured by the instantaneous frequency of the frame-line signal
(narrow band-pass around the pitch + Hilbert phase). That avoids the peak-picking
that was already found too noisy for --phase-lock adaptive, and unlike a windowed
autocorrelation it does not attenuate a once-per-turn modulation.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

SP = os.path.dirname(os.path.abspath(__file__))
STRIP = r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v12\wholeroll.npy"
CACHE = os.path.join(SP, "v12_profile.npy")
GEOM = r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v11\matched_walks.npz"
TWO_PI = 2.0 * math.pi
EPS = math.radians(0.3)

# ── 1. the v12 column profile (frame-line signal) ──
if os.path.exists(CACHE):
    prof = np.load(CACHE)
else:
    S = np.load(STRIP, mmap_mode="r")
    print("strip", S.shape, flush=True)
    acc = np.zeros(S.shape[1])
    for i in range(0, S.shape[0], 8):            # chunked to keep memory sane
        acc += np.asarray(S[i], float)
    prof = acc / len(range(0, S.shape[0], 8))
    np.save(CACHE, prof)
print("profile", prof.shape)

# ── 2. reconstruct the v12 block layout ──
d = np.load(GEOM, allow_pickle=True)
cx = np.asarray(d["cx_a"], float)[:, None]; cy = np.asarray(d["cy_a"], float)[:, None]
paths = d["paths"]; n = len(paths); nw = len(paths[0])
Ks = [paths[0][wi].shape[0] for wi in range(nw)]
Ms = [int(round(2 * EPS * Ks[i] / (TWO_PI - 2 * EPS))) for i in range(nw - 1)] + [0]
print(f"blocks {nw}; sum(K)={sum(Ks)} sum(M)={sum(Ms)} total={sum(Ks)+sum(Ms)} "
      f"(v12 strip was 211418)")
assert sum(Ks) + sum(Ms) == len(prof), "layout mismatch — abort"

# ── 3. build the v13 column map and resample the profile ──
out, starts = [], np.cumsum([0] + [Ks[i] + Ms[i] for i in range(nw)])
for wi in range(nw):
    K = Ks[wi]
    blk = prof[starts[wi]: starts[wi] + K]
    phi = np.linspace(EPS, TWO_PI - EPS, K)
    R = np.hypot(np.stack([paths[a][wi][:, 0] for a in range(n)]) - cx,
                 np.stack([paths[a][wi][:, 1] for a in range(n)]) - cy)
    rm = np.median(R, axis=0)
    ds = np.hypot(rm, np.gradient(rm, phi))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phi))])
    K2 = max(2, int(round(s[-1])))                       # 1.0 px per column
    phi_new = np.interp(np.linspace(0, s[-1], K2), s, phi)
    out.append(np.interp(phi_new, phi, blk))             # resampled block
    if Ms[wi]:                                           # keep the bridge columns
        out.append(prof[starts[wi] + K: starts[wi] + K + Ms[wi]])
prof13 = np.concatenate(out)
print(f"simulated v13 profile: {len(prof13)} cols (v12 {len(prof)})")


def local_pitch(p, f0=1 / 903.6, bw=0.28, smooth=400):
    """Instantaneous film-cell pitch via the analytic signal of the narrowly
    band-passed frame-line signal."""
    p = p - gaussian_filter1d(p, 1500)
    F = np.fft.rfft(p)
    f = np.fft.rfftfreq(len(p))
    F *= np.exp(-0.5 * ((f - f0) / (bw * f0)) ** 2)      # narrow Gaussian band-pass
    band = np.fft.irfft(F, n=len(p))
    ph = np.unwrap(np.angle(hilbert(band)))
    inst = TWO_PI / np.maximum(gaussian_filter1d(np.gradient(ph), smooth), 1e-9)
    return inst


pit12, pit13 = local_pitch(prof), local_pitch(prof13)
trim = 8000
a, b = pit12[trim:-trim], pit13[trim:-trim]
# compare as RELATIVE modulation (the strips have slightly different mean pitch)
ra, rb = a / np.median(a), b / np.median(b)
print(f"\n                     median pitch   relative modulation (p1..p99)   std")
print(f"  v12 (uniform phi)    {np.median(a):8.1f}   "
      f"{100*(np.percentile(ra,1)-1):+6.2f}% .. {100*(np.percentile(ra,99)-1):+6.2f}%"
      f"      {100*ra.std():.2f}%")
print(f"  v13 (arc length)     {np.median(b):8.1f}   "
      f"{100*(np.percentile(rb,1)-1):+6.2f}% .. {100*(np.percentile(rb,99)-1):+6.2f}%"
      f"      {100*rb.std():.2f}%")
print(f"\n  => modulation reduced {ra.std()/rb.std():.2f}x")

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharey=True)
ax[0].plot(np.arange(len(a)) + trim, a, lw=0.8, color="C3")
ax[0].axhline(np.median(a), color="k", lw=0.5)
ax[0].set_title(f"v12 (uniform-azimuth columns): local cell pitch  "
                f"[std {100*ra.std():.2f}%]")
ax[1].plot(np.arange(len(b)) + trim, b, lw=0.8, color="C0")
ax[1].axhline(np.median(b), color="k", lw=0.5)
ax[1].set_title(f"v13 (arc-length columns), same strip re-gridded: local cell pitch  "
                f"[std {100*rb.std():.2f}%]")
ax[1].set_xlabel("strip column")
plt.tight_layout(); plt.savefig(os.path.join(SP, "sim_v13_pitch.png"), dpi=110)
print("saved sim_v13_pitch.png")
