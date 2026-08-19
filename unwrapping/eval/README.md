# Film video evaluation — unrolled vs GT optical scan

Compares the unrolled CT film video against the ground-truth optical scan of the
**same physical Mickey reel**. The two are cross-modal (CT-derived positive vs
optical positive), different resolution / frame rate / field-of-view (GT keeps
sprockets + edge print), so the pipeline is **normalize → temporal-align →
spatial-register → score** with modality/alignment-robust metrics.

## Files
- `metrics.py` — NMI, gradient-correlation, SSIM, MS-SSIM, PSNR (self-contained
  numpy/scipy). Learned DISTS/LPIPS + no-reference BRISQUE/NIQE gated behind an
  optional `pyiqa` import (`HAS_PYIQA`); install pyiqa (Merlin7) to enable.
- `compare_videos.py` — Stage 0–3 pipeline + CLI. Outputs `scores.json`
  (aggregate + per-pair), `aligned_montage.png`, `dtw_path.png`.

## Validated invocation (2026-06-23)
The pred video is **rotated 90° and mirrored** relative to GT, and both need
cropping to the picture area (GT has perforations; pred has a margin/shading band):

```bash
python -m unwrapping.eval.compare_videos \
  --pred unwrapping/inr/results/arc_simple/wholeroll_v2/film_final_p902.mp4 \
  --gt   data/h265_1080p.mp4 \
  --out-dir unwrapping/eval/results/p902_vs_gt \
  --height 256 \
  --gt-crop 0.13,0.07,0.80,0.93 --pred-crop 0.0,0.12,1.0,0.98 \
  --pred-rot90 1 --pred-fliplr \
  --temporal linear            # or dtw
  --register gradphase         # gradient-domain phase-corr (fast, robust); or ecc/phasecorr/none
  --temporal-search 4          # refine each match within +/-4 frames (absorbs sync jitter)
  [--learned]                  # adds DISTS/LPIPS/BRISQUE/NIQE (needs pyiqa)
```

GT/pred crop boxes were read from gridded frames (`data/_gt_grid.png`,
`data/_pred_grid.png`). The orientation (rot90=1 + fliplr) was confirmed by the
title-card text reading correctly (`data/_titlecard_pair.png`).

## Status / findings
- **Stage 0 + 2 + 3 validated:** a hand-matched pair (the end "A CINE ART / The
  End / PICTURE" title card) registers to a (3,19)px shift and scores SSIM 0.54;
  under the full linear run the last two pairs hit **SSIM 0.985** — near-perfect
  on the clean static frame.
- **Temporal sync is near-1:1** (after dropping GT's ~4-frame leader): GT 229 vs
  pred 229, same direction; the proportional `--temporal linear` map lands the
  title card dead-on. DTW with the current gradient embedding does **not** beat
  linear (cross-modal cosine on 32×32 grad maps is too ambiguous on the slow,
  low-contrast animation).
- **Result profile:** clean static frame SSIM ~0.98; animated content ~0.30–0.34.
  Low content scores are part genuine (unroll shading/washout degrades busy
  frames) and part registration failure (only ~19% of pairs register <25px —
  FFT phase-correlation locks onto the shading/background on low-contrast frames).

## Cross-modal registration upgrade (DONE 2026-06-24)
`register.py` adds a gradient-LNCC **similarity transform** (translation+scale+
rotation) via torch autograd (`--register ecc`), and `compare_videos.py` adds a
fast **gradient-domain phase-correlation** translation (`--register gradphase`,
now default) plus a **temporal-search** refinement (`--temporal-search N`) that
re-picks each match within +/-N GT frames by best gradient-correlation.

- **Result (gradphase + search 4 vs the plain linear scaffold):** grad_corr
  **0.051 -> 0.106 (doubled)** across all regions (middle 0.059->0.105, end
  0.033->0.100); SSIM ~flat (0.283->0.295). The montage rows now visibly
  correspond (standing Mickey, bending scene, title card).
- **Why SSIM barely moves:** a joint temporal+spatial best-match search showed
  the optimal GT frame is within +/-5 of linear, but **even at the best match the
  cross-modal gradient correlation is only ~0.1-0.2** and the ECC similarity is
  tiny (scale ~1.02-1.12x, rot ~1deg). So spatial misalignment was minor; the
  residual is the **genuine cross-modal + unroll-quality gap** (pred shading/
  washout, low-contrast line art vs clean optical scan). That gap is the thing
  being measured, not an alignment artifact.
- **Metric profile:** clean static title frame SSIM ~0.98; animated content
  ~0.30 SSIM / ~0.10-0.12 grad_corr. Report NMI + grad_corr as the primary
  cross-modal numbers (SSIM/PSNR secondary, intensity-sensitive).

## BASELINE METRICS (full suite, 2026-06-24)
229 pairs, height 256, `--temporal linear --register gradphase --temporal-search 4
--learned`, run in nnunet env on Merlin7 (pyiqa 0.1.15). Result dir
`results/p902_vs_gt_learned/` (scores.json + aligned_montage.png).

| Metric | mean | dir | family |
|---|---|---|---|
| NMI | 1.024 | higher | cross-modal (primary) |
| gradient-corr | 0.106 | higher | cross-modal (primary) |
| DISTS | 0.306 | lower | learned FR, misalign-tolerant |
| LPIPS | 0.588 | lower | learned FR perceptual |
| MS-SSIM | 0.219 | higher | structural (secondary) |
| SSIM | 0.295 | higher | structural (secondary) |
| PSNR | 9.81 dB | higher | intensity (secondary) |
| VIF | 0.020 | higher | perceptual FR, info-fidelity ([0,1]) |
| MSAD | 0.235 | lower | mean-abs-diff ([0,1], intensity, weak) |
| **FID** | 288.9 | lower | set-level distribution distance |
| **VMAF** | 7.31 | higher | industry video FR ([0,100]) |
| BRISQUE (pred) | 13.03 | lower | NR, NSS-based (2012) |
| NIQE (pred) | 5.96 | lower | NR, NSS-based (2013) |
| CLIP-IQA (pred) | 0.452 | higher | NR, learned (modern ML) |
| MUSIQ (pred) | 40.33 | higher | NR, learned (modern ML) |

NR metrics computed on the un-warped pred frame. GPU run on gmerlin7 (~15 min
with FID/VIF) via `Scripts/slurm/slurm_eval_video.sh` (also exports
`aligned_pred.mp4`/`aligned_gt.mp4`). VMAF via static libvmaf ffmpeg
(`/data/user/li_k1/ffbin/...`) on the aligned pair, separate CPU sbatch ->
`vmaf.json`. **Reference bands:** FID 0=identical, <50 good, 100+ far apart
(biased high for our N=229 + cross-modal); VIF [0,1], good recon ~0.3-0.6, 0.02 =
near-zero shared info; VMAF ~93+ transparent, >80 good, <20 poor — 7.3 is very
low but its per-frame min 0 / max 100 shows the static title card scores ~perfect
while busy content ~0. **All four are cross-modal-depressed** (they assume
same-source distortion, not CT-vs-optical) — report them WITH that caveat, not
against streaming bars.

**Reading:** the *old* NSS no-reference metrics rate standalone quality as decent
(BRISQUE ~13 good; NIQE ~6 moderate), but the *modern learned* NR metrics are
more middling — CLIP-IQA 0.45 (good photos ~0.7-0.9) and MUSIQ 40 (good ~65-75,
KonIQ mean ~60) both say "moderate / below-average", reflecting the shading/
washout + low-contrast a human would mark down. Honest read: standalone quality
is "okay-to-moderate". The FR scores are modest and
dominated by the genuine cross-modal + unroll-quality gap (established: best
temporal+spatial match still only ~0.1-0.2 grad_corr, ECC scale ~1.0). DISTS 0.31
/ LPIPS 0.59 = moderate perceptual dissimilarity, not catastrophic. Report NMI +
grad_corr + DISTS as primary (modality/alignment-robust); SSIM/PSNR secondary.
Naive no-alignment SSIM (the strawman in the first scaffold run) was ~0.23 — the
aligned pipeline + pitch fix is the real baseline.

## Open work
1. **Pred-quality polish is the main lever** (not metrics tooling): the soft
   z-shading / washout bands genuinely cap the FR scores on busy frames.
   Per-slice/winding intensity normalization on the strip would raise the floor.
2. **ECC** is available but slow (~10s/pair) and only marginally better than
   gradphase here; use only if scale/rotation grows on another reel.
3. **pyiqa** is installed in the Merlin7 nnunet env (weights cached under
   `~/.cache/torch/hub`, incl. CLIP `RN50.pt` for CLIP-IQA); runs offline on a
   compute node.

## Running it (GPU, NEVER on a login node)
Compute on Merlin7 login nodes is **strictly forbidden** — always `sbatch`.
GPU jobs go to the `gmerlin7` cluster (the default `merlin7` cluster is CPU-only):
```bash
scp Scripts/slurm/slurm_eval_video.sh login001.merlin7.psi.ch:.../Scripts/slurm/
sbatch Scripts/slurm/slurm_eval_video.sh      # has #SBATCH --clusters=gmerlin7
squeue -M gmerlin7 -u $USER                    # monitor
```
`--learned` (DISTS/LPIPS/BRISQUE/NIQE/CLIP-IQA/MUSIQ) on CPU takes ~1h for 229
frames (MUSIQ/CLIP-IQA are transformers); on an A100 it is ~1-2 min.
```

---

# Frame matching (`frame_match.py`)

Pairs every GT optical-scan frame with the SAME film cell re-cut *continuously*
from the unrolled strip (not from the fixed video cells), for side-by-side
figures and a GT-locked standalone film. Runs LOCALLY (numpy/scipy, ~10 min);
needs the strip `wholeroll.npy` scp'd local. Output `results/frame_match_v9/`
(v9) and `results/frame_match_v12/` (v12): `pairs/*.png` (GT|ours|overlay|diff),
`side_by_side.mp4`, `film_stabilized.mp4`, `montage.png`, `matches.json`.

Pipeline: GT picture-crop `0.235,0.12,0.81,0.88` + leader-trim -> global ZOOM
calibration (fy/fx, ~1.06-1.11: GT shows ~11% more than one cell of arc) ->
cell-quantized **DP** (both sequences are cell-locked; GT advances 0/1/2 cells
per frame; motion-masked, contrast-normalized score; ±4-cell candidate span)
-> per-frame **GT-lock** (vertical/along-film = unsmoothed per-frame; horizontal
/across-film = GT-low-freq + self-registration high-freq crossover) -> render.
Key metric numbers vs `film_v9` baseline (same crop): grad_corr 0.205->0.58,
SSIM 0.36->0.57, MS-SSIM 0.39->0.67, PSNR 13->17.4dB (eval on the exact-1:1
`film_stabilized_gtsync.mp4`, `results/stab_vs_gt`). Full log: memory
`frame-matching-gt-pairs`.

# Stabilization (`stabilize_{horizontal,affine,flow}.py`)

**Diagnosis (load-bearing): the residual "wobble" in the stabilized video is
per-frame GEOMETRIC DEFORMATION from the unrolling, NOT translation** — the
picture breathes/rotates/warps ~8-12px at the edges (scale-y ±2.3%, scale-x
±1.5%, rotation ±0.76°, shear, + ~6px non-rigid), measured as the frame-to-frame
variation of a per-frame affine fit to the matched GT frame. The unroll flattens
a spiral at slightly inconsistent local scale per frame. So translation-only
stabilization can't work; the real fix is geometric consistency in the
emulsion-walk unrolling itself. All tools GT-anchored (GT is nearly still +
content-matched = the reliable reference) and run LOCALLY.

Fix ladder (increasing stillness, DECREASING faithfulness):
- `stabilize_horizontal.py --gt <gt.mp4>` -> `film_stabilized_hlock.mp4`:
  ITERATIVE translation lock (register to GT, remove the smooth offset, repeat;
  residual sway 10.3->1px verified). Faithful. (`--blur` GT-free fallback =
  unreliable, envelope too weak.)
- `stabilize_affine.py --mode affine --gt <gt.mp4>` -> `film_affine.mp4`:
  per-frame affine (scale/rotation/shear) to GT, iterated. Keeps OUR pixels;
  halves the wobble (scale-y 0.024->0.013). Most-still FAITHFUL-ish option.
- `stabilize_flow.py --gt <gt.mp4>` -> `film_flow.mp4`: dense **Farneback
  optical flow** to GT on EDGE maps (cross-modal; needs
  `pip install opencv-python-headless`, works in thesis env), iterated. STILLEST
  (scale-y 0.024->0.009), no artifacts, but REPROJECTS our pixels onto GT's
  per-pixel geometry => a **GT-STABILIZED VIEWING COPY**, not faithful: label as
  such and do NOT compute reconstruction-fidelity metrics on it.

CAUTION: metrics on the 1D column profile or consecutive-frame shift are
UNRELIABLE on this content-heavy film (polluted by the animation) — always
validate stabilization by RE-REGISTERING to the still GT. Full log: memory
`film-video-stabilization`.
