"""Head-to-head: de-shaded input vs FastDVDnet(s20) vs UDVD zero-shot(s15/25/40).
Two frames + a background zoom row."""
import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw

D = r"C:/Users/li_k1/M_thesis/data/video_restore_out"
IN = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
OUT = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad/compare_deep.png"

def read(p):
    r = iio.get_reader(p); f = [np.asarray(x).mean(2).astype(np.uint8) if np.asarray(x).ndim==3 else np.asarray(x) for x in r]
    r.close(); return np.stack(f)

vids = {"input (de-shaded)": read(IN),
        "FastDVDnet s20": read(f"{D}/fastdvdnet_s20.mp4"),
        "UDVD-zs s15": read(f"{D}/udvd_zs_s15.mp4"),
        "UDVD-zs s25": read(f"{D}/udvd_zs_s25.mp4"),
        "UDVD-zs s40": read(f"{D}/udvd_zs_s40.mp4")}
for k,v in vids.items(): print(k, v.shape)
H, W = vids["input (de-shaded)"].shape[1:]

def row(imgs, names, zoom=None):
    cells=[]
    for im,nm in zip(imgs,names):
        if zoom is not None:
            y0,x0,s=zoom; im=np.repeat(np.repeat(im[y0:y0+s,x0:x0+s],2,0),2,1)
        h,w=im.shape; c=np.full((h+30,w,3),20,np.uint8); c[30:]=np.repeat(im[...,None],3,2)
        pil=Image.fromarray(c); ImageDraw.Draw(pil).text((6,7),nm,fill=(255,230,60)); cells.append(np.asarray(pil))
    g=np.full((cells[0].shape[0],5,3),40,np.uint8); r=cells[0]
    for c in cells[1:]: r=np.concatenate([r,g,c],1)
    return r

rows=[row([vids[k][64] for k in vids], list(vids.keys())),
      row([vids[k][64] for k in vids], [f"{k} [zoom bg]" for k in vids], zoom=(40,470,200))]
Wm=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,Wm-r.shape[1]),(0,0)),constant_values=20) for r in rows]
g=np.full((6,Wm,3),40,np.uint8); full=rows[0]
for r in rows[1:]: full=np.concatenate([full,g,r],0)
Image.fromarray(full).save(OUT); print("saved", full.shape)
