import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw

IN  = r"C:/Users/li_k1/M_thesis/unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4"
S20 = r"C:/Users/li_k1/M_thesis/data/video_restore_out/fastdvdnet_s20.mp4"
OUT = r"C:/Users/li_k1/M_thesis/data/video_restore_out/compare_input_vs_fastdvdnet_s20.mp4"

def read(p):
    r=iio.get_reader(p); f=[np.asarray(x) for x in r]; r.close()
    return [x if x.ndim==3 else np.repeat(x[...,None],3,2) for x in f]
a,b=read(IN),read(S20); N=min(len(a),len(b)); H,W=a[0].shape[:2]; LAB=34
w=iio.get_writer(OUT,fps=15,quality=8)
for j in range(N):
    c=np.full((H+LAB,W*2+6,3),20,np.uint8); c[LAB:,:W]=a[j]; c[LAB:,W+6:]=b[j]
    im=Image.fromarray(c); d=ImageDraw.Draw(im)
    d.text((W//2-52,8),"DE-SHADED",fill=(255,230,60)); d.text((W+6+W//2-70,8),"FastDVDnet s20",fill=(80,220,255))
    w.append_data(np.asarray(im))
w.close(); print("wrote",OUT,N,"frames")
