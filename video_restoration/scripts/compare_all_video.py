import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw
IN=r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/newscan/stabilized/FLAGSHIP_borderlock_affine.mp4"
RD=r"C:/Users/li_k1/M_thesis/video_restoration/results/05_newscan_udvd"
OUT=RD+"/compare_newscan_all_sigmas.mp4"
srcs=[("ORIGINAL",IN),("UDVD s5",RD+"/udvd_zs_s5.mp4"),("UDVD s10",RD+"/udvd_zs_s10.mp4"),
      ("UDVD s15",RD+"/udvd_zs_s15.mp4"),("UDVD s25",RD+"/udvd_zs_s25.mp4")]
def read(p):
    r=iio.get_reader(p); f=[np.asarray(x) for x in r]; r.close()
    return [x if x.ndim==3 else np.repeat(x[...,None],3,2) for x in f]
vids=[read(p) for _,p in srcs]; N=min(len(v) for v in vids)
PW,PH,LAB=560,410,28
w=iio.get_writer(OUT,fps=15,codec='libx264',pixelformat='yuv420p',quality=8)
for j in range(N):
    cells=[]
    for (name,_),v in zip(srcs,vids):
        im=Image.fromarray(v[j]).resize((PW,PH))
        c=Image.new("RGB",(PW,PH+LAB),(20,20,20)); c.paste(im,(0,LAB))
        ImageDraw.Draw(c).text((6,7),name,fill=(255,230,60)); cells.append(np.asarray(c))
    gap=np.full((PH+LAB,4,3),40,np.uint8); rowimg=cells[0]
    for c in cells[1:]: rowimg=np.concatenate([rowimg,gap,c],1)
    w.append_data(rowimg)
w.close(); print("saved",OUT,"frames",N)
