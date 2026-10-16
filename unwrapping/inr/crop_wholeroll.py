"""Save full-res crops from the whole-roll strip at a few arc fractions."""
import sys
import numpy as np
from PIL import Image

npy, out_dir = sys.argv[1], sys.argv[2]
s = np.load(npy, mmap_mode="r")
W = s.shape[1]
for fr in [0.05, 0.35, 0.65, 0.9]:
    c0 = int(fr * W)
    c = np.asarray(s[:, c0:c0 + 3000], dtype=np.float32)
    lo, hi = np.percentile(c, [1, 99])
    d = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1)
    Image.fromarray((d * 255).astype(np.uint8)).save(f"{out_dir}/crop_{int(fr*100):02d}.png")
    print(f"crop {fr} cols {c0}-{c0+3000}", flush=True)
print("done")
