"""A content-INDEPENDENT view of the framing wobble: the cell-o-gram.

Cut the whole strip into film cells at one fixed pitch (exactly what
make_film_video --phase-lock global does) and stack them as the ROWS of an
image. The physical frame line between cells is a dark bar spanning the full
film width, so in this image it shows up as a near-vertical dark stripe.

  * framing perfectly steady  -> the stripe is dead STRAIGHT
  * framing creeping          -> the stripe WOBBLES, and the wobble is exactly
                                 what you see as the picture drifting in the video

Mickey moving between cells cannot fake this: the frame line is a fixed physical
feature of the film, not content.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

SP = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\v13_arclen_diag"
TWO_PI = 2.0 * math.pi
EPS = math.radians(0.3)

prof = np.load(os.path.join(SP, "v12_profile.npy"))
d = np.load(r"C:\Users\li_k1\M_thesis\unwrapping\inr\results\walk_dense_v11\matched_walks.npz",
            allow_pickle=True)
cx = np.asarray(d["cx_a"], float)[:, None]; cy = np.asarray(d["cy_a"], float)[:, None]
paths = d["paths"]; n = len(paths); nw = len(paths[0])
Ks = [paths[0][w].shape[0] for w in range(nw)]
Ms = [int(round(2 * EPS * Ks[i] / (TWO_PI - 2 * EPS))) for i in range(nw - 1)] + [0]
starts = np.cumsum([0] + [Ks[i] + Ms[i] for i in range(nw)])

out = []
for wi in range(nw):
    K = Ks[wi]; blk = prof[starts[wi]: starts[wi] + K]
    phi = np.linspace(EPS, TWO_PI - EPS, K)
    R = np.hypot(np.stack([paths[a][wi][:, 0] for a in range(n)]) - cx,
                 np.stack([paths[a][wi][:, 1] for a in range(n)]) - cy)
    rm = np.median(R, axis=0)
    ds = np.hypot(rm, np.gradient(rm, phi))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phi))])
    K2 = max(2, int(round(s[-1])))
    pn = np.interp(np.linspace(0, s[-1], K2), s, phi)
    out.append(np.interp(pn, phi, blk))
    if Ms[wi]:
        out.append(prof[starts[wi] + K: starts[wi] + K + Ms[wi]])
prof13 = np.concatenate(out)


def cellogram(p, tag):
    q = p - gaussian_filter1d(p, 1500)                  # isolate the frame-line signal
    lags = np.arange(850, 960)
    ac = np.array([np.dot(q[:-L], q[L:]) / (len(q) - L) for L in lags])
    k = int(np.argmax(ac)); y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    dd = y0 - 2 * y1 + y2
    pitch = lags[k] + (0.5 * (y0 - y2) / dd if abs(dd) > 1e-12 else 0.0)
    W = int(pitch); nc = int((len(q) - W) // pitch)
    img = np.empty((nc, W), np.float32)
    for i in range(nc):
        c0 = int(round(i * pitch))
        img[i] = np.interp(np.arange(W), np.arange(W),
                           q[c0:c0 + W] if c0 + W <= len(q) else q[-W:])
    # roll so the frame line sits near the middle, for display
    col = img.mean(0)
    img = np.roll(img, W // 2 - int(np.argmin(col)), axis=1)
    print(f"  {tag:22s} pitch {pitch:7.2f}  {nc} cells")
    return img, pitch


i12, p12 = cellogram(prof, "v12 uniform-azimuth")
i13, p13 = cellogram(prof13, "v13 arc-length")


def trace(img):
    """per-cell frame-line position (sub-px), relative to its own median"""
    W = img.shape[1]; c = W // 2
    win = img[:, c - 120:c + 120]
    x = np.arange(win.shape[1])
    w = np.clip(-win, 0, None) ** 2                    # weight the dark bar
    pos = (w * x).sum(1) / np.maximum(w.sum(1), 1e-9)
    return pos - np.median(pos)


t12, t13 = trace(i12), trace(i13)
print(f"\n  frame-line wobble  v12: ptp {np.ptp(t12):6.1f}px  std {t12.std():5.1f}px")
print(f"  frame-line wobble  v13: ptp {np.ptp(t13):6.1f}px  std {t13.std():5.1f}px")
print(f"  => {t12.std()/max(t13.std(),1e-9):.1f}x steadier")

fig, ax = plt.subplots(1, 4, figsize=(15, 8),
                       gridspec_kw={"width_ratios": [3, 1, 3, 1]})
for j, (img, t, tag, pit) in enumerate([(i12, t12, "v12  uniform-azimuth", p12),
                                        (i13, t13, "v13  arc-length", p13)]):
    a = ax[2 * j]
    v = np.percentile(np.abs(img), 99)
    a.imshow(img, cmap="gray", aspect="auto", vmin=-v, vmax=v,
             extent=[-img.shape[1] // 2, img.shape[1] // 2, len(img), 0])
    a.set_title(f"{tag}\ncells stacked; dark stripe = the film frame line",
                fontsize=10)
    a.set_xlabel("px within the cell"); a.set_ylabel("film cell #")
    a.set_xlim(-160, 160)
    b = ax[2 * j + 1]
    b.plot(t, np.arange(len(t)), lw=0.9, color="C3" if j == 0 else "C0")
    b.axvline(0, color="k", lw=0.5); b.set_ylim(len(t), 0); b.set_xlim(-70, 70)
    b.set_title(f"wobble\nptp {np.ptp(t):.0f}px", fontsize=9)
    b.set_xlabel("px")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cellogram_AB.png"), dpi=115)
print("saved cellogram_AB.png")
