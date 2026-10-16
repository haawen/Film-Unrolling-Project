# Emulsion-walk unwrapping algorithm — sample package

This is the core algorithm that traces each spiral winding of the film
through one CT cross-section, plus one real sample data chunk to run/debug
it against. Verified working standalone from this folder as-is.

## What it does (from the script's own docstring)

Given a CT slice segmented into background / film-base / emulsion classes,
recover the spiral film windings by walking each one directly, rather than
assuming a fixed geometric model:

1. **Seed**: march outward along one clean ray and count class-2 (emulsion)
   runs at least `--min-thick` px thick — the middle of each run is one seed
   point, one per winding. The count comes straight from the data (no
   center-based counting, no gap-infill guessing).
2. **Walk**: from each seed, follow its emulsion band tangentially in both
   directions with a predictor-corrector: step along the local tangent, then
   snap onto the class-2 centroid within a small perpendicular window
   (`--snap-max`, capped below half the winding spacing so it can't jump onto
   a neighbour). Coasts across dashed/broken segments of emulsion
   (`--coast-max`).

Why this approach (vs. a polar r(θ)-about-a-center model, which was tried
first and abandoned): it has no dependence on a center or eccentricity (the
walk is purely local), it can't wander onto a neighbouring winding (the
perpendicular snap is capped), it stays on the emulsion sublayer even where
film bases touch, and it isn't broken by the radial "pinch" connections that
defeated an earlier geodesic-based approach.

This is the core per-slice/per-winding algorithm only. In production it's
one stage of a larger pipeline (seeding consensus across all windings,
z-sequential tracking across hundreds of slices, outlier healing, arc-length
column rendering) — not included here, ask if you want that too.

## Requirements

Python 3, with: `numpy`, `scipy`, `matplotlib`, `h5py`. No `torch` needed —
this package only pulls in the parts of the codebase the walk itself uses.

## Run it

```bash
cd walk_algo_share
python -m unwrapping.inr.walk_emulsion \
    --vol sample_data/volume_1595-1614.h5 \
    --probs sample_data/volume_1595-1614_Probabilities.h5 \
    --out-dir out
```

(Run from this folder, or otherwise make sure this folder is on `PYTHONPATH`
so `unwrapping.inr.walk_emulsion` resolves — it's a package-relative import,
not a standalone script.)

Confirmed output on this sample chunk:
```
20 slices; center=(1529,1551); seam=56.5deg; seed_az=254.0deg; r[488,1396]
  37 windings seeded (radii 619..1381)
  32 windings after dedup (min-sep 8.0)
  path lens [2032, 3956, 4093, 4234, 4497, 4604]...; strip (20, 65024). Done -> out
```
`out/` gets 5 PNGs: `walk_ct.png` / `walk_seg.png` (every walked path overlaid
on the raw CT / segmentation, the main thing to look at first),
`walk_zoom.png` (a close-up crop), `strip_overview.png` / `strip_crop.png`
(the unrolled unwrap result for this one 20-slice chunk).

Useful flags to try while debugging (all have defaults, see `--help` /
the argparse block at the bottom of `walk_emulsion.py` for the complete,
documented list): `--min-thick`, `--snap-max`, `--jump-max`, `--coast-max`,
`--smooth-px`, `--seed-mode {film,emulsion}`.

## Sample data

`sample_data/volume_1595-1614.h5` (raw CT, key `volume`) and
`..._Probabilities.h5` (per-voxel class softmax, key `exported_data`,
argmax'd internally into background/film/emulsion labels) — one real
20-slice chunk of the CT-scanned 1930s film reel this project unrolls. This
is the exact chunk the script's own docstring names as its prototyping
example. Real segmentation, so it has the dashes/patchiness/noise the
algorithm was actually designed to handle — a synthetic clean spiral
wouldn't exercise the same failure modes.

## Files

```
unwrapping/
  __init__.py, inr/__init__.py         (package markers, empty)
  center_detection.py                  spool-center estimation (unedited)
  inr/walk_emulsion.py                 *** the algorithm + its CLI ***
  inr/_walk_helpers.py                 3 small helpers the walk calls
sample_data/
  volume_1595-1614.h5
  volume_1595-1614_Probabilities.h5
```

Only `walk_emulsion.py`'s algorithm code (the ~32KB of actual walking logic)
and `center_detection.py` are unedited, verbatim copies from the working
codebase. `_walk_helpers.py` is new to this package: it holds 3 small,
self-contained functions the walk calls that, in the main codebase, live
inside three much larger *unrelated* files (an abandoned earlier unwrapping
approach, and a separate INR-based method) — rather than ship those whole
files just for one function each, their bodies are copied verbatim
(unmodified logic) into this one small file, and only `walk_emulsion.py`'s
import line was changed to point at it (see the NOTE at the top of the file).
Nothing about how the algorithm itself works was touched.
