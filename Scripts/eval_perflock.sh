#!/bin/bash
# Fair GT-referenced comparison of the perf-locked render against the render it
# is meant to replace. Runs LOCALLY (the GT scan lives here, and the whole eval
# chain in unwrapping/eval has always run local).
#
# CROP MATCHING IS THE WHOLE POINT and is easy to get wrong -- see the crop trap
# in the film-video-stabilization notes. The two predictions have DIFFERENT
# framing, so they need different --pred-crop to show the same physical area:
#
#   perf-locked   already cropped to the ISO aperture (10.26 mm across), so it
#                 needs no across crop
#   film_newscan1 the full film width, 2361 strip rows; the aperture is rows
#                 415-1935 about the film centre, i.e. x 0.176-0.820
#
# Both keep y 0.12-0.88 so the along-film extent matches the GT crop's ~5.8 mm.
#
#   bash Scripts/eval_perflock.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/c/Users/li_k1/AppData/Local/miniforge3/envs/thesis/python.exe
R=unwrapping/inr/results
GT=data/h265_1080p.mp4
OUT=unwrapping/eval/results

run () {   # name  video  pred-crop
  echo
  echo "=== $1 ==="
  "$PY" -m unwrapping.eval.compare_videos \
      --pred "$2" --gt "$GT" \
      --out-dir "$OUT/perflock_eval_$1" \
      --gt-crop 0.21,0.12,0.81,0.88 \
      --pred-crop "$3" \
      --temporal linear --register gradphase \
      --height 256
}

run baseline_phaselock "$R/newscan_render1/film_newscan1_ph.mp4" 0.176,0.12,0.820,0.88
run perflock_perf      "$R/newscan_perflock_perf/film_perflock_perf.mp4"           0.0,0.12,1.0,0.88
run perflock_frameline "$R/newscan_perflock_frameline/film_perflock_frameline.mp4" 0.0,0.12,1.0,0.88

echo
echo "=== summary ==="
for d in baseline_phaselock perflock_perf perflock_frameline; do
  echo "--- $d"
  "$PY" -c "
import json,sys
try:
    s=json.load(open(r'$OUT/perflock_eval_$d/scores.json'))
    for k in ('grad_corr','ssim','ms_ssim','psnr','nmi','lpips','dists','niqe','brisque'):
        if k in s: print(f'  {k:10s} {s[k]:.4f}')
except Exception as e: print('  ',e)
"
done
