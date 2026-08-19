"""Before/after of ORIGINAL vs geo-normalized FULL strip, each shown as the video
will render it (own 1/99 percentile + invert). Stacked before/after per crop."""
import numpy as np
from scipy import ndimage
from PIL import Image

A = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll.npy"
B = r"C:/Users/li_k1/M_thesis/unwrapping/inr/results/walk_dense_v9/wholeroll_geonorm.npy"
OUT = r"C:/Users/li_k1/AppData/Local/Temp/claude/c--Users-li-k1-M-thesis/417e7200-4070-4b8f-950c-0d5eb90692c8/scratchpad"

def lohi(mm):
    W = mm.shape[1]
    s = np.asarray(mm[:, np.arange(2000, W-2000, 40)], np.float32)
    return np.percentile(s, [1, 99])

ma, mb = np.load(A, mmap_mode='r'), np.load(B, mmap_mode='r')
la, ha = lohi(ma); lb, hb = lohi(mb)
g8 = lambda a: np.repeat((np.clip(a,0,1)*255).astype(np.uint8)[...,None],3,2)
H = ma.shape[0]; gap = np.full((6, 1150, 3), 30, np.uint8)

for name, x0 in [("mid_b", 120000), ("late", 170000), ("early", 40000)]:
    va = 1 - np.clip((np.asarray(ma[:, x0:x0+1150], np.float32)-la)/(ha-la+1e-6),0,1)
    vb = 1 - np.clip((np.asarray(mb[:, x0:x0+1150], np.float32)-lb)/(hb-lb+1e-6),0,1)
    Image.fromarray(np.concatenate([g8(va), gap, g8(vb)], 0)).save(f"{OUT}/cmp_{name}.png")  # top=orig, bottom=cleaned
    print(f"{name}: done")
print("saved")
