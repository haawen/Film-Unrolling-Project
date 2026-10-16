"""Classical motion-aware TEMPORAL fusion on the stabilized de-shaded video.
Faithful (no training, no hallucination): a temporal BILATERAL filter — for each
pixel, average +-N neighbor frames weighted by REGIONAL intensity similarity to the
current frame. Static regions (background: weights ~1 across frames) get denoised /
dust removed; moving regions (character: neighbor weight ~0) fall back to the
current frame -> no ghosting. Optional unsharp afterwards. Exploits that dust is a
per-frame temporal outlier and the cel background recurs across cells.
"""
import numpy as np
import imageio.v2 as iio
from scipy import ndimage
from PIL import Image, ImageDraw

SRC = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
OUTV = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/temporal_fused.mp4"
OUTC = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/temporal_fused_compare.mp4"
OUTM = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad/temporal_montage.png"

N     = 3       # +-N neighbor frames
SIM   = 0.10    # similarity sigma (smaller = more motion-protective, less fusion)
SMOOTH= 2.0     # spatial blur of the squared diff (regional, not per-pixel noise)
TDECAY= 2.5     # temporal weight falloff (frames)
USHARP_AMT, USHARP_SIG = 0.6, 1.4   # unsharp mask

r = iio.get_reader(SRC)
frames = [np.asarray(f, np.float32).mean(axis=2)/255.0 for f in r]  # grayscale [0,1]
r.close()
F = np.stack(frames); T, H, W = F.shape
print(f"loaded {T} frames {H}x{W}", flush=True)

def fuse(t):
    ft = F[t]
    acc = np.zeros_like(ft); wsum = np.zeros_like(ft)
    for k in range(max(0,t-N), min(T,t+N+1)):
        d2 = ndimage.gaussian_filter((F[k]-ft)**2, SMOOTH)
        w = np.exp(-d2/(2*SIM*SIM)) * np.exp(-((k-t)/TDECAY)**2)
        acc += w*F[k]; wsum += w
    return acc/np.maximum(wsum,1e-6)

out = np.empty_like(F)
for t in range(T):
    o = fuse(t)
    o = np.clip(o + USHARP_AMT*(o - ndimage.gaussian_filter(o, USHARP_SIG)), 0, 1)  # unsharp
    out[t] = o
    if t % 50 == 0: print(f"  frame {t}/{T}", flush=True)

# write fused video + a side-by-side compare (de-shaded | temporally fused+sharpened)
g8 = lambda a: (np.clip(a,0,1)*255).astype(np.uint8)
w1 = iio.get_writer(OUTV, fps=15, quality=8)
w2 = iio.get_writer(OUTC, fps=15, quality=8)
LAB=34; mont=[]; midx=[40,90,140,195]
for t in range(T):
    w1.append_data(g8(out[t]))
    a, b = g8(F[t]), g8(out[t])
    canv = np.full((H+LAB, W*2+6, 3), 20, np.uint8)
    canv[LAB:,:W]=np.repeat(a[...,None],3,2); canv[LAB:,W+6:]=np.repeat(b[...,None],3,2)
    im=Image.fromarray(canv); d=ImageDraw.Draw(im)
    d.text((W//2-52,8),"DE-SHADED",fill=(255,230,60)); d.text((W+6+W//2-70,8),"+TEMPORAL FUSE+SHARP",fill=(80,220,255))
    fr=np.asarray(im); w2.append_data(fr)
    if t in midx: mont.append(fr)
w1.close(); w2.close()
Image.fromarray(np.concatenate(mont,0)).save(OUTM)
print(f"wrote {OUTV}\n{OUTC}\nmontage {OUTM}")
