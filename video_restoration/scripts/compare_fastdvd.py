"""Compare de-shaded input vs FastDVDnet outputs (sigma sweep). Montage of a few
frames x {input, s12, s20, s30} + a background zoom to judge grain reduction."""
import numpy as np, imageio.v2 as iio
from scipy import ndimage
from PIL import Image, ImageDraw

IN  = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
D   = r"C:/Users/li_k1/M_thesis/data/video_restore_out"
OUT = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad"

def read(path):
    r = iio.get_reader(path)
    fr = [np.asarray(f).mean(axis=2).astype(np.uint8) if np.asarray(f).ndim==3 else np.asarray(f) for f in r]
    r.close(); return np.stack(fr)

vids = {"DE-SHADED (input)": read(IN),
        "FastDVDnet s12": read(f"{D}/fastdvdnet_s12.mp4"),
        "FastDVDnet s20": read(f"{D}/fastdvdnet_s20.mp4"),
        "FastDVDnet s30": read(f"{D}/fastdvdnet_s30.mp4")}
for k,v in vids.items(): print(k, v.shape)
T = min(v.shape[0] for v in vids.values())
H, W = vids["DE-SHADED (input)"].shape[1:]

def label_row(imgs, names, zoom=None):
    cells=[]
    for im, nm in zip(imgs, names):
        if zoom is not None:
            y0,x0,s = zoom; im = im[y0:y0+s, x0:x0+s]
            im = np.repeat(np.repeat(im, 2, 0), 2, 1)   # 2x nearest upscale
        h,w = im.shape
        c = np.full((h+30, w, 3), 20, np.uint8); c[30:] = np.repeat(im[...,None],3,2)
        d = ImageDraw.Draw(Image.fromarray(c))  # measure only
        pil = Image.fromarray(c); ImageDraw.Draw(pil).text((6,7), nm, fill=(255,230,60))
        cells.append(np.asarray(pil))
    gap = np.full((cells[0].shape[0], 5, 3), 40, np.uint8)
    row = cells[0]
    for c in cells[1:]: row = np.concatenate([row, gap, c], 1)
    return row

rows = []
for j in [64, 140]:
    rows.append(label_row([vids[k][j] for k in vids], list(vids.keys())))
# background zoom (upper-right flat wall area) on frame 64
rows.append(label_row([vids[k][64] for k in vids], [f"{k} [zoom bg]" for k in vids], zoom=(40, 470, 200)))
Wmax = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0,0),(0,Wmax-r.shape[1]),(0,0)), constant_values=20) for r in rows]
gap = np.full((6, Wmax, 3), 40, np.uint8)
full = rows[0]
for r in rows[1:]: full = np.concatenate([full, gap, r], 0)
Image.fromarray(full).save(f"{OUT}/fastdvd_compare.png")
print("saved fastdvd_compare.png", full.shape)
