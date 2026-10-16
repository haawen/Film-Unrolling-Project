"""Export an mp4 to a folder of %05d.png frames (for UDVD-S training input)."""
import cv2, os, sys
inp, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(inp); i = 0
while True:
    r, f = cap.read()
    if not r:
        break
    cv2.imwrite(os.path.join(outdir, '%05d.png' % i), f); i += 1
print('exported', i, 'frames to', outdir)
