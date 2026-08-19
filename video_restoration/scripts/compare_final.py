"""Final 4-way: input(de-shaded) | FastDVDnet s20 | UDVD-zs s25 | UDVD-S trained.
Two frames + background zoom + a side-by-side video (input vs UDVD-S)."""
import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw

R = r"C:/Users/li_k1/M_thesis/video_restoration/results"
IN = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
OUTM = r"C:/Users/li_k1/M_thesis/video_restoration/results/comparisons/compare_all_denoisers.png"
OUTV = r"C:/Users/li_k1/M_thesis/video_restoration/results/comparisons/compare_input_vs_udvd_s.mp4"

def read(p, gray=True):
    r = iio.get_reader(p); out=[]
    for x in r:
        a=np.asarray(x)
        out.append(a.mean(2).astype(np.uint8) if (gray and a.ndim==3) else a)
    r.close(); return np.stack(out)

vids = {"input (de-shaded)": read(IN),
        "FastDVDnet s20": read(f"{R}/02_fastdvdnet/fastdvdnet_s20.mp4"),
        "UDVD zero-shot s25": read(f"{R}/03_udvd_zeroshot/udvd_zs_s25.mp4"),
        "UDVD-S (trained on ours)": read(f"{R}/04_udvd_s/udvd_s_trained.mp4")}
for k,v in vids.items(): print(k, v.shape)
H,W = vids["input (de-shaded)"].shape[1:]

def row(imgs,names,zoom=None):
    cells=[]
    for im,nm in zip(imgs,names):
        if zoom is not None:
            y0,x0,s=zoom; im=np.repeat(np.repeat(im[y0:y0+s,x0:x0+s],2,0),2,1)
        h,w=im.shape; c=np.full((h+30,w,3),20,np.uint8); c[30:]=np.repeat(im[...,None],3,2)
        pil=Image.fromarray(c); ImageDraw.Draw(pil).text((6,7),nm,fill=(255,230,60)); cells.append(np.asarray(pil))
    g=np.full((cells[0].shape[0],5,3),40,np.uint8); r=cells[0]
    for c in cells[1:]: r=np.concatenate([r,g,c],1)
    return r
rows=[row([vids[k][64] for k in vids],list(vids.keys())),
      row([vids[k][140] for k in vids],list(vids.keys())),
      row([vids[k][64] for k in vids],[f"{k} [zoom bg]" for k in vids],zoom=(40,470,200))]
Wm=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,Wm-r.shape[1]),(0,0)),constant_values=20) for r in rows]
g=np.full((6,Wm,3),40,np.uint8); full=rows[0]
for r in rows[1:]: full=np.concatenate([full,g,r],0)
Image.fromarray(full).save(OUTM); print("saved montage", full.shape)

# side-by-side video input vs UDVD-S
a,b = vids["input (de-shaded)"], vids["UDVD-S (trained on ours)"]; N=min(len(a),len(b)); LAB=34
w=iio.get_writer(OUTV,fps=15,quality=8)
for j in range(N):
    c=np.full((H+LAB,W*2+6,3),20,np.uint8); c[LAB:,:W]=np.repeat(a[j][...,None],3,2); c[LAB:,W+6:]=np.repeat(b[j][...,None],3,2)
    im=Image.fromarray(c); d=ImageDraw.Draw(im)
    d.text((W//2-52,8),"DE-SHADED",fill=(255,230,60)); d.text((W+6+W//2-70,8),"UDVD-S (trained)",fill=(80,220,255))
    w.append_data(np.asarray(im))
w.close(); print("saved video", OUTV)
