import sys, numpy as np, imageio.v2 as iio
src, dst = sys.argv[1], sys.argv[2]
r = iio.get_reader(src); frames = [np.asarray(f) for f in r]; r.close()
w = iio.get_writer(dst, fps=15, codec='libx264', pixelformat='yuv420p', quality=8)
for f in frames: w.append_data(f)
w.close(); print("re-encoded", len(frames), "frames ->", dst)
