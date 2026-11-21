"""Export a wholeroll.npy strip (float, 0..1) as a full-resolution 16-bit TIFF, in the
same format as render/wholeroll_tw4_16bit.tif: uint16 = round(v*65535), BigTIFF,
deflate, 512x512 tiles, JSON metadata in ImageDescription.

  python Scripts/export_strip_tiff.py <strip.npy> <out.tif> [--note "..."]
"""
import argparse
import json

import numpy as np
import tifffile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip"); ap.add_argument("out")
    ap.add_argument("--z0", type=int, default=40)
    ap.add_argument("--pitch", type=float, default=1130.0)
    ap.add_argument("--note", default=None)
    a = ap.parse_args()
    s = np.load(a.strip, mmap_mode="r")
    H, W = s.shape
    u = np.empty((H, W), np.uint16)
    for i in range(0, H, 128):
        u[i:i + 128] = np.round(np.clip(np.nan_to_num(s[i:i + 128], nan=0.0), 0, 1) * 65535)
    meta = {"shape": [H, W], "content": "unrolled 16mm film strip, CT reconstruction",
            "axis0": f"across film ({H} rows, 1 per CT slice, z = row + {a.z0})",
            "axis1": "along film (arc length, 1 px = 1 px of film)",
            "value": "float = uint16 / 65535; film is a NEGATIVE, invert to view",
            "picture_band_rows": "480-1890", "frame_pitch_px": a.pitch,
            "play_direction": "reversed (last column first)"}
    if a.note:
        meta["note"] = a.note
    tifffile.imwrite(a.out, u, bigtiff=True, compression="zlib", tile=(512, 512),
                     description=json.dumps(meta), metadata=None)
    with tifffile.TiffFile(a.out) as t:
        back = t.pages[0].asarray()
    assert np.array_equal(back, u), "read-back mismatch"
    print(f"wrote {a.out}: {u.shape} uint16, read-back identical; "
          f"p1/p50/p99 {np.percentile(u[::8, ::64], [1, 50, 99])}")


if __name__ == "__main__":
    main()
