#!/bin/bash
# Pull a small working slab out of the 2.4 GB full-width strip so the perf
# detector can be iterated on locally instead of via a cluster round-trip per
# change. Also computes the per-row AIR FRACTION over the whole strip, which is
# the discriminator the first detect_perfs run should have used: the
# periodic-at-frame-pitch amplitude does not separate the perf rows from the
# picture band (picture frame lines are periodic at the same pitch, baseline
# 0.10 vs peak 0.18), whereas "what fraction of this row is a hole full of air"
# separates them completely.
#
#   sbatch Scripts/slurm/slurm_perf_slab.sh
#
#SBATCH --job-name=perf_slab
#SBATCH --partition=hourly
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/perf_slab_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

R="$PROJ/unwrapping/inr/results"
OUT="$R/newscan_perfs"
mkdir -p "$OUT"

python -u - <<'PY'
import numpy as np
R = "/data/user/li_k1/M_thesis/unwrapping/inr/results"
OUT = f"{R}/newscan_perfs"
s = np.load(f"{R}/newscan_render1/wholeroll.npy", mmap_mode="r")
Z, W = s.shape
print("strip", s.shape, s.dtype, flush=True)

# ~7 frames from three places along the reel, full film width.
for tag, c0 in (("in", 5_000), ("mid", 120_000), ("out", 240_000)):
    c1 = min(W, c0 + 8000)
    np.save(f"{OUT}/slab_{tag}.npy", np.asarray(s[:, c0:c1], np.float32))
    print(f"slab_{tag}: cols {c0}-{c1}", flush=True)

# global intensity histogram -> air / film levels
samp = np.asarray(s[::4, ::16], np.float32).ravel()
pcts = np.percentile(samp, [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99])
print("intensity percentiles:", np.round(pcts, 1), flush=True)

# per-row air fraction, at a threshold midway between the air and film modes
air = np.percentile(samp, 1)
film = np.percentile(samp, 75)
thr = 0.5 * (air + film)
print(f"air~{air:.1f} film~{film:.1f} thr={thr:.1f}", flush=True)
frac = np.empty(Z)
for i in range(Z):
    row = np.asarray(s[i, ::4], np.float32)
    frac[i] = (row < thr).mean()
np.save(f"{OUT}/airfrac.npy", frac)
np.save(f"{OUT}/levels.npy", np.array([air, film, thr]))
print("airfrac done", flush=True)
PY

ls -la "$OUT"
