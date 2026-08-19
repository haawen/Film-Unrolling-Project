# v13 arc-length columns — evidence bundle (2026-07-27)

Diagnosis of the residual wobble in `film_v12.mp4`, and validation of the fix.
Everything here was produced **locally** from the v12 deliverable + its geometry
cache — no cluster time, no re-walk, no re-render yet.

Inputs used:
- geometry `results/walk_dense_v11/matched_walks.npz` (= the v12 fit: 36 windings,
  500 anchors, z-heal 6px) — pulled from Merlin7.
- strip `results/walk_dense_v12/wholeroll.npy` (1184 x 211418).

---

## The finding

**The residual wobble is a parameterization error, not a fit error.**

`unroll_walk_wholeroll.py` puts each winding's strip columns on a uniform
**azimuth** grid, so every column covers `ds/dphi = hypot(r, dr/dphi)` px of film.
`r` swings by the winding's eccentricity (~80 px ptp), so the film length per
column swings with it — coherently across all 36 windings, **once per turn**.

What was ruled out, measured on the same cache:

| candidate | measured | verdict |
|---|---|---|
| in-plane jitter of the fitted curves | **0.23 px rms** (median winding) | already smooth |
| film tilt vs z, `sqrt(1+(dr/dz)^2)` | 1.0003 → **0.06%** row-scale variation | ruled out |
| winding-to-winding scale at the joins | 0.23%, worst step 8 px | ruled out |
| **film-px per column within a turn** | **0.914 … 1.060 (±5%)** | **the cause** |

The 0.23 px rms figure also explains why the earlier `--z-smooth-anchors`
σ ∈ {1.5, 3} sweep found nothing to gain: there was nothing left to smooth.

---

## Figures

| file | what it shows |
|---|---|
| `arclen_diag.png` | the geometry prediction: local scale vs fraction of turn, per winding; cumulative along-film position error; and that the modulation is **coherent across windings** (= eccentricity, once per turn) |
| `localpitch.png` | local film-cell pitch measured on the **delivered v12 strip** — periodic, one-winding period |
| `sim_v13_pitch.png` | v12 vs the same strip **re-gridded to arc length**: the regular oscillation disappears |
| `quant.png` | per-winding decomposition into the once-per-turn component vs everything else |
| `sim_frames_creep.png` | **the key plot** — where each cell's content lands vs its fixed-pitch slot, within one turn |
| `AB_cells_fullres.png` / `AB_cells_half.png` | 5 consecutive cells, v12 above / v13 below, red guide line at the same position in every cell |
| `cells_v12_fullres.png`, `cells_v13_fullres.png` | the same montages without the guide line, for close inspection |

---

## Numbers

Local cell pitch on the v12 strip, decomposed per winding:

```
                       once-per-turn amplitude    everything else (std)
  v12 uniform-azimuth          3.98%                     1.12%
  v13 arc-length               1.50%                     0.96%
  => breathing reduced 2.7x; broadband unchanged (real film + estimator noise)
```

Framing creep, cutting cells at a fixed pitch (what `make_film_video
--phase-lock global` does), one turn of winding 8:

```
  v12 uniform-azimuth   pitch 895.74   framing offset: ptp 85.7px  std 30.3px
  v13 arc-length        pitch 903.76   framing offset: ptp  8.1px  std  3.2px
  => creep reduced 10.5x  (9.6% -> 0.9% of frame width)
```

The v12 curve is a clean arch — content drifts **86 px out and back within 5
frames**, repeating every turn (~33 times through the film).

Code validation of `arc_equalize()` against the real cache
(`scripts/test_arc_equalize.py`):

```
 wi     K -> K2   scale err BEFORE   AFTER    arc kept   xz-align
  3   4126 ->   4127      +7.31%       +0.04%   100.000%    0.000deg
 15   5578 ->   5584      +4.55%       +0.05%   100.000%    0.000deg
 34   8365 ->   8360      +3.65%       +0.14%   100.000%    0.000deg
z-heal on the non-uniform arc grid: 20px jump -> 1.48px residual (OK)
```

`xz-align 0.000deg` is the one that matters: the phi-per-column map is computed
once and shared by every z, so column *c* still means the same azimuth at every
z — the cross-z alignment `resample_phi` exists to provide is preserved exactly.

---

## Caveats (read before trusting the numbers)

- `sim_v13.py` / `sim_frames.py` re-grid the **already-rendered** strip rather
  than re-sampling the CT. A true v13 render avoids that double interpolation so
  it should be at least this good — but part of the residual 1.50% may be
  simulation artifact rather than real.
- The ~1% broadband pitch variation is **not** touched, and probably should not
  be: some of it is genuine film.
- The fitted curves are unchanged, so this cannot help per-slice intensity
  banding or the join artifacts.
- Residual spikes in the re-gridded pitch all sit within ~3k columns of a
  **winding join** — a separate issue (seam-bridge / block boundary), not the
  breathing.

## What did NOT work (don't repeat)

`FAILED_cellogram_AB.png` / `scripts/FAILED_cellogram.py` — an attempt to show
the wobble content-independently by stacking all ~235 cut cells as image rows,
expecting the physical frame line to appear as a vertical stripe that wobbles.
It does not work: the high-pass used (sigma 1500, wider than the 904px cell) does
not isolate the frame line, so the stripe never forms and the tracer locks onto
picture content instead. Its numbers (237px ptp, "1.3x steadier") are garbage.
Registering cells against each other only works over a FEW consecutive cells,
where the animation has barely changed — which is why the winding-8 measurement
in `sim_frames_creep.png` is valid and a whole-strip version is not.

---

## The code

`unwrapping/inr/unroll_walk_wholeroll_v13.py` — a **fork**, so the live
`unroll_walk_wholeroll.py` (being edited by the other session for the full-z
work) is untouched. New `arc_equalize()`; `--no-arc-columns` reverts to the v12
behaviour for an A/B; `--px-per-col` sets the film px per column (default 1.0,
constant over the whole roll now rather than per winding).

Also needed inside the fork: `z_heal_windings` now reads each winding's own phi
grid instead of assuming it is uniform, and the seam-bridge width is derived from
the arc length of the missing wedge rather than from the uniform-phi ratio.

Proposed A/B (not yet run — identical to the v12 command except the script):

```bash
D=unwrapping/inr/results/walk_dense_v13
ln -s .../walk_dense_v11/walk_anchors.npz $D/walk_anchors.npz
python -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir $D \
    --use-cached-anchors --ref-mode track --min-coverage 50 --z-heal-px 6 \
    --seam-exclude-deg 0.3 --seam-bridge
python unwrapping/inr/crop_wholeroll.py $D/wholeroll.npy $D
python -m unwrapping.inr.make_film_video $D/wholeroll.npy $D/film_v13.mp4 \
    --phase-lock global --reverse --invert --height 720
```

`scripts/` holds every diagnostic used above, runnable as-is.
