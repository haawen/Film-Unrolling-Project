import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw

IN = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/newscan/stabilized/FLAGSHIP_borderlock_affine.mp4"
DN = r"C:/Users/li_k1/M_thesis/video_restoration/results/05_newscan_udvd/udvd_zs_s25.mp4"
OUTM = r"C:/Users/li_k1/M_thesis/video_restoration/results/05_newscan_udvd/compare_newscan_udvd_s25.png"
OUTV = r"C:/Users/li_k1/M_thesis/video_restoration/results/05_newscan_udvd/compare_newscan_udvd_s25.mp4"

def read(p):
    r=iio.get_reader(p); f=[np.asarray(x).mean(2).astype(np.uint8) if np.asarray(x).ndim==3 else np.asarray(x) for x in r]
    r.close(); return np.stack(f)
a,b = read(IN), read(DN); N=min(len(a),len(b)); H,W=a.shape[1:]
print("orig",a.shape,"denoised",b.shape)

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
# pick two frames + a flat-region zoom
f1,f2=80,150
rows=[row([a[f1],b[f1]],["ORIGINAL","UDVD zero-shot s25"]),
      row([a[f2],b[f2]],["ORIGINAL","UDVD zero-shot s25"]),
      row([a[f1],b[f1]],["ORIGINAL [zoom]","UDVD s25 [zoom]"],zoom=(int(H*0.15),int(W*0.55),220))]
Wm=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,Wm-r.shape[1]),(0,0)),constant_values=20) for r in rows]
g=np.full((6,Wm,3),40,np.uint8); full=rows[0]
for r in rows[1:]: full=np.concatenate([full,g,r],0)
Image.fromarray(full).save(OUTM); print("saved montage",full.shape)

LAB=34; w=iio.get_writer(OUTV,fps=15,quality=8)
for j in range(N):
    c=np.full((H+LAB,W*2+6,3),20,np.uint8); c[LAB:,:W]=np.repeat(a[j][...,None],3,2); c[LAB:,W+6:]=np.repeat(b[j][...,None],3,2)
    im=Image.fromarray(c); d=ImageDraw.Draw(im)
    d.text((W//2-40,8),"ORIGINAL",fill=(255,230,60)); d.text((W+6+W//2-60,8),"UDVD zero-shot s25",fill=(80,220,255))
    w.append_data(np.asarray(im))
w.close(); print("saved video",OUTV)
