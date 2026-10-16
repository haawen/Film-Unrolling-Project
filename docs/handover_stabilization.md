# Film stabilization — handover brief

## What this footage is

A 16 mm film reel that was **never physically unrolled**. The reel was CT-scanned
while still wound, the film layers were traced through the scan volume, and each
frame was reconstructed computationally. There is no camera and no scan of a
flattened film.

Consequences that matter for your work:

- **No lens distortion, no rolling shutter, no motion blur.** Turn those options
  off in any stabilizer that offers them.
- **The instability is not camera shake.** See below — this is the single most
  important point.
- 1 video frame = 1 film cell, in order. No pulldown, no field interlacing.
- The film is an early animation short, originally silent (~16–18 fps). The
  export frame rate is arbitrary and carries no meaning.

## The file to work from

| file | use |
|---|---|
| `film_stabilized_gtsync.mp4` | **START HERE.** 822×720, 26 fps, 233 frames. Every pixel is ours, ungeometrically-altered. |
| `film_affine.mp4` | Reference only — **do not stabilize this one.** It has already had a per-frame affine warp fitted to a reference scan. Stabilizing it would stack a second correction on a first. |

Both files are already the right way round: correct play direction, correct
orientation, and tonally positive (the film itself is a negative; the inversion
is applied).

## The nature of the instability — please read

Standard position-only stabilization will barely help here, and that is not a
tuning problem. The residual motion is **per-frame geometric deformation**, not
translation. Measured against a reference scan, frame to frame:

| component | magnitude (std) |
|---|---|
| scale x | ~1.2 % |
| scale y | ~1.3 % |
| rotation | ~0.8° |
| non-rigid residual | a few px, varies within the frame |

Each frame is reconstructed slightly differently in size and shape, because the
flattening scale depends on where that frame sat in the spiral. So:

- A **translation-only** tracker (one-point track, position-only warp) removes a
  minority of the visible motion.
- **Affine or perspective** (scale + rotation + shear) is the minimum useful
  model. In our own tests it roughly halved the wobble.
- A **mesh / subspace warp** (AE Warp Stabilizer "Subspace Warp", Mocha planar
  with mesh, Resolve stabilizer at "Perspective") reaches the non-rigid part.
- Please note we already tried a dense optical-flow warp to a reference. It was
  the stillest by measurement but looked visibly worse — softer — and was
  rejected. Stillness alone is not the goal; image quality matters equally.

## Tracking features — what is and is not reliable

- **DO** track the printed picture border / frame lines. They are inside the
  well-reconstructed area and are the most trustworthy fiducial.
- **DO NOT** track the sprocket holes if you see them in any wider version of
  this footage. They are reconstructed from extrapolated geometry and do not
  move with the picture; we tested this and confirmed they are unusable as
  fiducials. They will actively mislead a tracker.

## Known defects — please do not try to fix these

They are reconstruction artifacts, not stabilization problems, and chasing them
will distort good frames:

- A **vertical band of slightly different brightness/texture** toward one side,
  in roughly the first third of the reel. Cause is still unexplained on our end.
- A patch of **stronger wobble around the 4-second mark**. It comes from the
  scan geometry in that part of the spiral.
- The **outermost turn of film is missing** (~4 frames of length). Nothing to
  recover.
- Some **brightness variation along the reel** is genuine film density, not
  flicker. A gentle deflicker is fine if it helps; please tell us if you apply
  one, and keep it separable from the stabilization.

## What we need back

1. **Same frame count, same order, no retiming.** Do not drop, duplicate,
   blend, or interpolate frames. Frame ↔ film-cell correspondence is used in our
   evaluation, and any retiming invalidates it.
2. **Lossless or near-lossless output** (ProRes 422 HQ / 4444, DNxHR HQX, or a
   PNG/TIFF sequence). Not a second H.264 pass.
3. **A note on what you did**: which transform model (translation / affine /
   perspective / mesh), which tool, and whether any reference footage was used.
4. **Per-frame transform data exported**, if your tool can (AE keyframe data,
   Mocha track export, Resolve stabilization data). This is genuinely valuable to
   us — it tells us how each frame was deformed, which feeds back into fixing the
   reconstruction itself.
5. Please keep cropping to the minimum needed to hide edge fill.

## One request

If a warp is fitted to any external reference footage, say so explicitly. Such a
result is fine to look at, but it is no longer a faithful reconstruction — the
picture has been reshaped toward something else — and we have to label it
differently in a scientific write-up.

---

# Appendix — working from the raw strip

If you have someone who can write Python, you can skip our frame cutting
entirely and do it yourself from `wholeroll_tw4.npy`. This is the highest-quality
route: full resolution, no compression, and you choose where every frame is cut.
None of the following is inferable from the file, so please read it.

## The file

| | |
|---|---|
| format | NumPy `.npy`, `float32`, C-order, 2.4 GB |
| shape | `(2361, 259297)` |
| axis 0 | **across the film** (film width, 2361 samples, one per CT slice) |
| axis 1 | **along the film** (arc length; 1 column = 1 px of film, uniform) |
| values | ~0.036–1.000, reconstructed X-ray attenuation. **The film is a negative — invert for display.** |

It is ONE CONTINUOUS RIBBON of film, not frames. Load it with
`np.load(path, mmap_mode="r")` and slice; do not read it into RAM whole.

## Constants you need

| | |
|---|---|
| frame pitch | **1130.0 px** along the film (~229 frames total) |
| picture band | **rows 480–1890** (everything else is perforations/margin) |
| low perforation row | rows ~160–420 |
| high perforation row | rows ~1950–2210 |
| display normalisation | `lo = 0.1487`, `hi = 0.9882` (1st/99th pct), then **invert** |
| orientation | `np.rot90(window, 1)` — no mirror/flip for this scan |
| play direction | **reversed** — the last frame in the strip is the FIRST on screen |
| row → CT slice | `z = row + 40` (contiguous, no gaps) |

## Minimal frame cutter

```python
import numpy as np
s = np.load("wholeroll_tw4.npy", mmap_mode="r")     # (2361, 259297)
Z0, Z1, PITCH = 480, 1890, 1130.0

def frame(k, phase=0.0):
    c0 = int(round(phase + k * PITCH))
    f = np.asarray(s[Z0:Z1, c0:c0 + int(PITCH)], np.float32)
    f = 1.0 - np.clip((f - 0.1487) / (0.9882 - 0.1487), 0, 1)   # normalise + invert
    return np.rot90(f, 1)          # -> 1130 tall (along film) x 1410 wide (across film)

# play order: k = 228, 227, ..., 0
```

## Traps, in order of how much time they will cost you

1. **The cell phase is NOT free — and the obvious way to find it is wrong.**
   Autocorrelating the column mean of the WHOLE strip lands you half a pitch out,
   every frame split across two. The perforation rows are periodic at exactly the
   frame pitch and dominate that signal. **Compute the phase signal from the
   picture band rows only (`s[480:1890]`)**, and then confirm by eye that frames
   are centred rather than split. This cost us a full render to discover.
2. **The frame pitch is not perfectly uniform.** 417 synthetic blended columns
   were inserted at 33 winding boundaries where the film's seam falls, so local
   pitch varies by a fraction of a percent and those columns are interpolated,
   not measured. A per-frame search (rather than a fixed pitch) handles this and
   is what we ended up doing.
3. **Only rows 480–1890 are real geometry.** The perforation rows and margins are
   rendered from extrapolated geometry — they look plausible and are not
   trustworthy. Do not track the sprocket holes.
4. **Columns are uniform in film arc length**, not in angle, so along-film
   distance is linear in column index. That is already corrected for you.
5. The **outermost end of the film** (highest column indices) includes the reel
   tongue and is partly junk; we cut it from our video.

## If you would rather start from our solution

We already solved a per-frame cut position for all 233 frames by matching each
one against a reference scan (sub-pixel across film, cell-quantized along it).
Ask and we will send `video_bounds.npy` — the locked cut position per frame —
which you can use as a starting point and refine, instead of solving it cold.
