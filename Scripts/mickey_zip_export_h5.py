"""Export z-ranges of Mickey_merged_rigid_nobin-tif.zip to per-slice HDF5 and
(optionally) scp them to the cluster in bounded batches.

WHY a separate naming/dir: this reconstruction is NOT the old stitch. It is
3738x3769 (old: 3063x3062) and ~1.245x FINER in-plane (film radial range
r768-1678 vs the old r617-1347; winding spacing ~26px vs 21px). Its z index is
its own. Mixing the two frames would silently corrupt any render -- the same class
of bug as the H<->W seg transpose -- so files keep the zip's own name
`Mickey_merged_{z:04d}.h5` and land in their OWN directory. Nothing existing is
touched or replaced.

z structure measured by Scripts/diag_zip_zsurvey.py (film area + components):
    z <~60        packaging (the "useless" slices)
    z ~75-200     film margin (full film area)
    z ~200-460    LOW PERFORATION row   (~18% area dip, shattered arcs)
    z ~470-1975   picture band          (= what the old stitch kept)
    z ~1990-2250  HIGH PERFORATION row  (~18% area dip)
    z ~2260-2350  film margin
    z >~2360      packaging

Usage:
  python Scripts/mickey_zip_export_h5.py --zip data/Mickey_merged_rigid_nobin-tif.zip \
      --z-ranges 40-540,1930-2400 --stage <local staging dir> \
      [--scp-target li_k1@login001.merlin7.psi.ch:/data/user/li_k1/M_thesis/01_Mickey_sprockets] \
      [--batch 50] [--gzip 1]
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import h5py
import numpy as np
import tifffile as tf


def parse_ranges(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.append((int(a), int(b if b else a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--z-ranges", required=True,
                    help="e.g. 40-540,1930-2400 (inclusive, zip indices)")
    ap.add_argument("--stage", required=True, help="local staging dir")
    ap.add_argument("--scp-target", default="",
                    help="user@host:/abs/dir . If set, each batch is scp'd and "
                         "then DELETED locally, so staging stays bounded.")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--gzip", type=int, default=1,
                    help="h5 gzip level (0 = none). 1 gives ~15%% on this CT.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    pat = re.compile(r"_(\d+)\.tif$", re.I)
    zmap = {int(pat.search(n).group(1)): n for n in zf.namelist() if pat.search(n)}
    want = []
    for a, b in parse_ranges(args.z_ranges):
        want += [k for k in range(a, b + 1) if k in zmap]
    want = sorted(set(want))
    print(f"{len(zmap)} slices in zip; exporting {len(want)} "
          f"(z {want[0]}..{want[-1]})")
    if args.dry_run:
        return

    os.makedirs(args.stage, exist_ok=True)
    if args.scp_target:
        host, _, rdir = args.scp_target.partition(":")
        subprocess.run(["ssh", host, f"mkdir -p {rdir}"], check=True)

    kw = dict(compression="gzip", compression_opts=args.gzip) if args.gzip else {}
    t0 = time.time()
    done = 0
    for i in range(0, len(want), args.batch):
        chunk = want[i:i + args.batch]
        for k in chunk:
            a = tf.imread(io.BytesIO(zf.read(zmap[k])))
            p = os.path.join(args.stage, f"Mickey_merged_{k:04d}.h5")
            with h5py.File(p, "w") as f:
                d = f.create_dataset("image", data=a, **kw)
                d.attrs["source"] = os.path.basename(args.zip)
                d.attrs["source_member"] = zmap[k]
                d.attrs["zip_z"] = k
                d.attrs["frame"] = ("Mickey_merged_rigid_nobin reconstruction; "
                                    "NOT the old 3063x3062 stitch frame")
        if args.scp_target:
            # one scp per batch; retry once (long transfers on a shared login node)
            files = [os.path.join(args.stage, f"Mickey_merged_{k:04d}.h5")
                     for k in chunk]
            for attempt in (1, 2):
                r = subprocess.run(["scp", "-q"] + files + [args.scp_target])
                if r.returncode == 0:
                    break
                print(f"  scp failed (attempt {attempt}), retrying", flush=True)
                time.sleep(5)
            else:
                sys.exit("scp failed twice -- aborting, nothing deleted")
            for p in files:
                os.remove(p)
        done += len(chunk)
        el = time.time() - t0
        print(f"  [{done}/{len(want)}] z{chunk[0]}-{chunk[-1]} | "
              f"{el / 60:.1f} min elapsed, "
              f"ETA {el / done * (len(want) - done) / 60:.0f} min", flush=True)

    if args.scp_target:
        shutil.rmtree(args.stage, ignore_errors=True)
    print(f"Done: {done} slices in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
