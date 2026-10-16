"""GT (left) vs ours (right) side-by-side. GT is leader-trimmed + picture-cropped +
grayscale to match the frame_match framing; ours = the (denoised) film_affine.
Labels: 'GT' and 'ours'. Outputs an H.264 mp4."""
import numpy as np, imageio.v2 as iio
from PIL import Image, ImageDraw

GT   = r"C:/Users/li_k1/M_thesis/data/h265_1080p.mp4"
OURS = r"C:/Users/li_k1/M_thesis/video_restoration/results/06_newscan_affine_vs_gt/ours_affine_udvd_s10_raw.mp4"
OUT  = r"C:/Users/li_k1/M_thesis/video_restoration/results/06_newscan_affine_vs_gt/GT_vs_ours_affine_s10.mp4"
TRIM = 4
CROP = (0.235, 0.12, 0.81, 0.88)   # GT picture area (verified vs film_affine)
PH   = 620                          # panel height

def read_gray(p):
    r = iio.get_reader(p); f=[np.asarray(x).mean(2) if np.asarray(x).ndim==3 else np.asarray(x) for x in r]; r.close(); return f

def crop(im):
    h,w=im.shape; x0,y0,x1,y1=CROP; return im[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]

def panel(im, label, w):
    im=(np.clip(im,0,255)).astype(np.uint8)
    pil=Image.fromarray(im).resize((w,PH))
    c=Image.new("RGB",(w,PH+30),(20,20,20)); c.paste(pil.convert("RGB"),(0,30))
    ImageDraw.Draw(c).text((6,7),label,fill=(255,230,60)); return np.asarray(c)

gt=read_gray(GT)[TRIM:]; ours=read_gray(OURS)
N=min(len(gt),len(ours)); print("GT(trimmed)",len(gt),"ours",len(ours),"-> N",N)
# panel widths preserve each source's aspect at height PH
gw=int(PH*crop(gt[0]).shape[1]/crop(gt[0]).shape[0])
ow=int(PH*ours[0].shape[1]/ours[0].shape[0])
w=iio.get_writer(OUT,fps=15,codec='libx264',pixelformat='yuv420p',quality=8)
gap=np.full((PH+30,6,3),40,np.uint8)
for j in range(N):
    L=panel(crop(gt[j]),"GT",gw); R=panel(ours[j],"ours",ow)
    w.append_data(np.concatenate([L,gap,R],1))
w.close(); print("saved",OUT,"frames",N)
