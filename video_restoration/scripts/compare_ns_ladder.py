import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw
IN=r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/newscan/stabilized/FLAGSHIP_borderlock_affine.mp4"
RD=r"C:/Users/li_k1/M_thesis/video_restoration/results/05_newscan_udvd"
def read(p):
    r=iio.get_reader(p); f=[np.asarray(x).mean(2).astype(np.uint8) if np.asarray(x).ndim==3 else np.asarray(x) for x in r]; r.close(); return np.stack(f)
V={"ORIGINAL":read(IN),"s5":read(RD+"/udvd_zs_s5.mp4"),"s10":read(RD+"/udvd_zs_s10.mp4"),
   "s15":read(RD+"/udvd_zs_s15.mp4"),"s25":read(RD+"/udvd_zs_s25.mp4")}
H,W=V["ORIGINAL"].shape[1:]
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
names=list(V.keys())
rows=[row([V[k][150] for k in names],names),
      row([V[k][150] for k in names],[k+" [zoom]" for k in names],zoom=(int(H*0.15),int(W*0.55),200))]
Wm=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,Wm-r.shape[1]),(0,0)),constant_values=20) for r in rows]
g=np.full((6,Wm,3),40,np.uint8); full=rows[0]
for r in rows[1:]: full=np.concatenate([full,g,r],0)
Image.fromarray(full).save(RD+"/compare_newscan_sigma_ladder.png"); print("saved",full.shape)
