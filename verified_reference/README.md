# Virtual Unrolling of Motion-Picture Film — Master's Thesis Deliverable

**GitHub checkout:** this directory contains code and documentation only.
The `Data/` and `Results/` paths below refer to the separately distributed
handoff and are intentionally absent from GitHub.

**Combined handoff layout.** This verified reference sits inside the main handoff. To avoid duplicate files, the ready-to-train Dataset501 and Dataset502 are at `../Segmentation/nnUNet_data/nnUNet_raw/`, and the final strip is at `../unwrapping/inr/results/newscan/render/wholeroll_tw4.npy`. `Code/env.sh` and `Code/slurm/eval_01_final_vs_gt.sh` use those locations. The raw full-width CT is excluded and must be supplied separately at `Data/3_full_width_ct_scan/ct_slices/` for CT-dependent reruns. The original Dataset502 CT and Ilastik probability exports are included at `../data/01_Mickey_3d/`. See the [main README](../README.md) for the complete combined handoff instructions.


Kai Wen Li · Paul Scherrer Institute (PSI) · October 2026

A tightly wound, chemically degraded 16 mm film roll (an early Mickey Mouse cartoon)
was scanned with synchrotron X-ray tomography **without unwinding it**. This
deliverable turns that 3D CT volume into a flat, playable film:

```
CT volume ──► 3-class segmentation ──► emulsion walk ──► flat strip ──► film frames
            (nnU-Net 3D, Dice 0.973)   (one ordered curve    (arc-length      (video, compared
                                        per physical winding)  columns)         with an optical scan)
```

Everything needed to reproduce the reported results is in three folders (plus this README, [`requirements.txt`](requirements.txt) and [`VERIFICATION.md`](VERIFICATION.md), which records the re-runs performed before hand-over):

| Folder | Contents |
|---|---|
| [`Data/`](#3-data) | Segmentation training data, trained models, the full-width CT scan (segmented, walk-ready), cached walk geometry, the optical ground-truth scan, synthetic-benchmark inputs |
| [`Code/`](#10-code-map) | One Python package (`virtual_unrolling`), segmentation scripts, and one SLURM job script per pipeline step |
| [`Results/`](#9-results-and-where-every-reported-number-comes-from) | The requested videos, the final strip, every score file behind the reported numbers, figures |

Contents: [1 Requested videos](#1-requested-videos) · [2 Installation](#2-installation) ·
[3 Data](#3-data) · [4 Segmentation: train + evaluate](#4-segmentation-train-and-evaluate) ·
[5 Segment the scan](#5-segment-the-ct-scan) · [6 Scan parameters](#6-scan-parameters-seam-and-picture-band) ·
[7 Emulsion walk](#7-emulsion-walk-on-the-segmented-data) ·
[8 Videos and evaluation](#8-videos-and-evaluation-against-the-optical-scan) ·
[9 Results ↔ numbers](#9-results-and-where-every-reported-number-comes-from) ·
[10 Code map](#10-code-map) · [11 Parameters that matter](#11-parameters-that-matter) ·
[12 Known issues](#12-known-issues-and-caveats) · [13 Compute](#13-compute-requirements)

**Pipeline at a glance** (all job scripts in [`Code/slurm/`](Code/slurm/), submitted from the deliverable root):

| Step | Job script | Needed to reproduce? |
|---|---|---|
| Train / evaluate the segmentation model | `seg_01` … `seg_04` | evaluation runs on the shipped models; training is optional |
| Segment the CT scan | `seg_05_segment_scan.sh` | no — the segmentation is shipped |
| Measure seam angle + picture band | `param_01_seam_and_picture_band.sh` | no — re-derives two constants |
| Emulsion walk (5 z-ranges) | `walk_01_walk_range.sh b0…tail` | no — the walk caches are shipped |
| Merge + render final strip + plain video | `walk_02_merge_and_render.sh` | from the shipped caches (~1 h) |
| Early-version render | `walk_02_merge_and_render.sh early` | from the shipped caches (~1 h) |
| GT frame alignment, scores, GT-free border lock, stillness | `eval_01_final_vs_gt.sh` | runs on the shipped final strip |

---

## 1. Requested videos

All in [`Results/0_requested_videos/`](Results/0_requested_videos/). Film is a negative; all videos are already inverted to a positive and play in the correct direction.

| File | What it is |
|---|---|
| `1a_early_version_emulsion_walk__full_width.mp4` | **Early version of the walking algorithm** — the "Emulsion walk" stage of the ablation, rendered on the full-width scan from the same walk as the final model: consensus seeding, predictor–corrector walk, rescue, profile tracking, z-healing, but strip columns still on a uniform **angle** grid and a single sample on the emulsion centre line (the final model adds arc-length columns and transverse averaging). 229 cells, no GT used. `walk_02_merge_and_render.sh early`. |
| `1b_early_version_emulsion_walk__picture_area_scan__as_in_presentation.mp4` | The same stage exactly as shown in the presentation's ablation (slide "Reconstruction improves with refinement", row "Emulsion walk"). That ablation was run on an earlier, picture-area-only reconstruction of the same roll, which is not part of this deliverable; this file is included as a reference and cannot be rebuilt from `Data/`. |
| `2a_final_model_WITH_gt_frame_alignment__per_frame_affine.mp4` | **Final model, GT used for frame alignment** — 233 frames. Each optical frame is paired with the same film cell, the cell is re-cut from the strip at the position that best matches it (`frame_match`), then one affine warp per frame is fitted to the optical frame (`stabilize_affine`). This is the scored video (grad-corr 0.788, MS-SSIM 0.857, LPIPS 0.143). |
| `2b_final_model_WITH_gt_frame_alignment__cut_positions_only.mp4` | Same, but **only the cut positions** come from the GT (no warp): every pixel is our reconstruction, unwarped. |
| `3a_final_model_NO_gt__plain_render.mp4` | **Final model, GT NOT used** — the final strip cut into 229 cells at a constant pitch with the strip's own phase lock (both perforation rows visible). |
| `3b_final_model_NO_gt__border_locked.mp4` | **Final model, GT NOT used, stabilised** — the printed picture border is tracked *in the strip itself* and each cell is locked to it across the film; along the film the strip's own phase lock (229 cells, 2 % margin so the border stays visible). Stillness 71 % of the perforation-locked optical scan. |
| `extra_presentation_clip__optical_GT_left__full_width_render_right.mp4` | The clip shown in the presentation (optical scan left, plain render right). |

Side-by-side with the optical scan: `Results/2_final_model_full_width/frame_match/side_by_side.mp4`.

---

## 2. Installation

Python 3.12, CUDA 12.4. All results were produced on PSI Merlin7 (A100-80GB GPUs for nnU-Net; CPU nodes with 64–360 GB RAM for walking/rendering).

```bash
conda create -n nnunet python=3.12 -y
conda activate nnunet
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt          # pinned versions, see requirements.txt

# every session / before every sbatch:
cd /path/to/M_thesis_deliverable
source Code/env.sh                       # sets THESIS_ROOT, PYTHONPATH, nnU-Net + MONAI paths, $WORK, $D_SCAN, $GT_VIDEO
```

[`requirements.txt`](requirements.txt) lists all libraries with the exact versions used. `pyiqa` (LPIPS/DISTS/NIQE…) is optional and downloads its model weights on first use.

**nnU-Net custom trainer.** Training and inference use the trainer class `nnUNetTrainerProgress` (= the default nnU-Net trainer, 250 epochs, plus a progress bar and a checkpoint every 25 epochs). It must be registered with the installed nnU-Net once (the job scripts do this automatically):
```bash
cp Code/segmentation/nnunet_custom_trainer.py \
   $(python -c "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/custom_trainer.py
```

**Running jobs.** Every step has a script in [`Code/slurm/`](Code/slurm/) with the exact command lines used. Submit from the deliverable root: `sbatch Code/slurm/<script>.sh`. The `#SBATCH` lines are for Merlin7 (GPU jobs need `--clusters=gmerlin7`); adapt partition names and the conda line in `Code/slurm/_activate.sh` for another cluster. Each script header documents its arguments, runtime and outputs. Outputs go to `$WORK` (default `work/`), never into `Data/` or `Results/`. The scripts are plain bash and also run without SLURM (`bash Code/slurm/<script>.sh …`) if the machine has the resources.

---

## 3. Data

| Full-width CT scan | |
|---|---|
| slices (z) | 2361 (z 40–2400): both perforation rows, the picture band and the film margins |
| slice size | 3738 × 3769 px |
| film windings / spacing | 34 / 26 px |
| emulsion thickness | ~5 px |
| spiral seam (film outer end) | 353° |
| picture band | z ≈ 470–1975 (clean z 600–1930); perforation rows below/above |
| frame pitch in the strip | 1130 px |
| frames | 229 (outermost partial turn rejected) |

Segmentation classes everywhere: **0 = air, 1 = film base, 2 = emulsion** (the image-bearing layer).

The **segmentation training data** were annotated on an earlier, picture-area-only reconstruction of the same roll (3063 × 3062 px slices, ~1.25× coarser sampling). The trained model is applied to the full-width scan at its native resolution without retraining or new labels.

```
Data/
├── 1_segmentation_training/
│   ├── nnUNet_raw/
│   │   ├── Dataset502_MickeyScroll3D/   25 annotated 3D sub-volumes (20 slices each), NIfTI
│   │   │                                imagesTr/Mickey3D_<z0>_0000.nii.gz + labelsTr/Mickey3D_<z0>.nii.gz
│   │   └── Dataset501_MickeyScroll/     50 independently annotated 2D slices (.tif), "original 2D" baseline
│   └── splits/                          nnU-Net fold definitions actually used (splits_final.json per dataset)
│                                        Dataset502 fold 0: test = Mickey3D_1263, 1547, 1669, 1774, 1847
├── 2_trained_models/
│   ├── nnUNet_results/Dataset502_MickeyScroll3D/nnUNetTrainerProgress__nnUNetPlans__3d_fullres/
│   │        plans.json, dataset.json, fold_0/checkpoint_final.pth (THE selected model), progress.png,
│   │        training log, fold_0/validation/*.nii.gz (predictions on the 5 test volumes)
│   ├── nnUNet_results/Dataset501_…__2d/, Dataset503_…__2d/    2D comparison models (same layout)
│   └── monai_results/Dataset502_MickeyScroll3D/{UNet3D,SwinUNETR}/fold_0/checkpoints/best.pt (+ validation/)
├── 3_full_width_ct_scan/
│   ├── ct_slices/                       2361 CT slices, Mickey_merged_<z>.h5, key 'image' (uint16)
│   ├── segmentation_nnunet3d_picture_band/  nnU-Net 3D segmentation of z470–1969
│   │                                    (75 chunks on the 20-slice grid 470+20k, labels only, 1.4 GB)
│   └── walk_caches/                     b0, b1, b2, b3, tail_dense .npz = the five walks merged into the final strip
├── 4_optical_ground_truth/
│   ├── optical_scan_h265_1080p.mp4      conventional optical scan of the same film AFTER physical unrolling
│   └── optical_scan_perflocked_picture.mp4  same scan re-cut on its own perforations (stillness reference)
└── 5_synthetic_inputs/                  source videos for the synthetic benchmark (Big Buck Bunny 1080p, …)
```

**File formats ("ready to use").** CT slices are one HDF5 per z-slice (`load_ct`). Segmentations are 20-slice chunks `volume_<z0>-<z1>_Probabilities.h5` with dataset `exported_data` of shape (Z, H, W, 3); every consumer uses `argmax(axis=-1)`. To keep the deliverable small they are stored as **uint8 one-hot** (gzip), produced by [`compact_segmentation.py`](Code/segmentation/compact_segmentation.py) from the float softmax and verified voxel-identical; the walk code reads them unchanged. Walk caches (`walk_anchors.npz` format): per walked slice the z index, spool centre, seam and the list of traced emulsion curves (x, y).

**Where the raw CT lives.** `Data/3_full_width_ct_scan/ct_slices` (54 GB) is a symbolic link into the thesis working directory on Merlin7 (`/data/user/li_k1/M_thesis/01_Mickey_sprockets`), to avoid duplicating it. When copying the deliverable elsewhere, dereference it (`rsync -aL` / `cp -rL`) or point the link at your copy. The per-slice files were exported from the original reconstruction (TIFF stack `Mickey_merged_rigid_nobin-tif.zip`, 64 GB, on Merlin7).

**Provenance.** The segmentation is the nnU-Net 3D prediction (selected model, `--disable_tta`). Its float softmax had been deleted after walking except for z1590–1969; z470–1589 was regenerated for this deliverable with the shipped model and the identical command (`seg_05_segment_scan.sh`). The walk caches are the original walks; walking b0 again on the regenerated segmentation gives the same geometry (median difference 0.01 px, [`VERIFICATION.md`](VERIFICATION.md)). The source chunks of Dataset501/502 (ilastik exports) and the scripts that converted them are not shipped, since both datasets are shipped prebuilt; the sources remain on Merlin7 under `/data/user/li_k1/M_thesis/` (`01_Mickey_3d/` for Dataset502, `01_Mickey_hdf_subset/` for Dataset501).

---

## 4. Segmentation: train and evaluate


Segmentation done using "Ilastik" Software.

The selected model is **nnU-Net v2, 3d_fullres, Dataset502, fold 0** (20 training / 5 held-out sub-volumes). Shipped: `Data/2_trained_models/nnUNet_results/Dataset502_MickeyScroll3D/…/fold_0/checkpoint_final.pth`.

### 4.1 Train

```bash
sbatch Code/slurm/seg_01_preprocess.sh            # plan + preprocess Dataset502, install the fold definition
sbatch Code/slurm/seg_02_train_nnunet3d.sh        # train fold 0 (~20 h on one A100) -> $WORK/nnUNet_results
```
Equivalent raw commands (after `source Code/env.sh`, trainer registered as in §2):
```bash
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres
cp Data/1_segmentation_training/splits/Dataset502_MickeyScroll3D_splits_final.json \
   $nnUNet_preprocessed/Dataset502_MickeyScroll3D/splits_final.json
export nnUNet_results=$WORK/nnUNet_results        # do not overwrite the shipped model
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerProgress --npz
```

The four comparison models of the segmentation table:
```bash
sbatch Code/slurm/seg_01_preprocess.sh 501 503    # Dataset503 (2D slices of 502) is built first, then preprocessed
sbatch Code/slurm/seg_03_train_baselines.sh nnunet2d_matched    # Dataset503, 2d
sbatch Code/slurm/seg_03_train_baselines.sh nnunet2d_orig       # Dataset501, 2d
sbatch Code/slurm/seg_03_train_baselines.sh unet3d              # MONAI, 250 epochs, patch 16x192x192
sbatch Code/slurm/seg_03_train_baselines.sh swinunetr
```

### 4.2 Evaluate

```bash
sbatch Code/slurm/seg_04_evaluate.sh                    # evaluates the SHIPPED models
MODELS_FROM=work sbatch Code/slurm/seg_04_evaluate.sh   # evaluates your retrained models
```
This runs `Code/segmentation/fair_compare.py`: every model is scored on identical voxels of the held-out 3D volumes (the 2D models' slice predictions are reassembled into volumes). Dice / IoU / precision / recall are computed per volume for the two film classes (film base, emulsion) and averaged over classes and volumes; air is not included in the average. See §12 for which volumes each model is scored on. Output: `$WORK/fair_comparison/fair_comparison_results.json` + `figures/`. Reported result: [`Results/1_segmentation/fair_comparison_results.json`](Results/1_segmentation/).

| Model | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| **nnU-Net 3D** (selected) | **0.9728** | **0.9475** | 0.9718 | **0.9738** |
| nnU-Net 2D (matched) | 0.9717 | 0.9454 | **0.9725** | 0.9709 |
| UNet3D (MONAI) | 0.9447 | 0.8969 | 0.9428 | 0.9467 |
| SwinUNETR (MONAI) | 0.9350 | 0.8788 | 0.9243 | 0.9459 |
| nnU-Net 2D (original, 50 slices) | 0.7485 | 0.6280 | 0.9234 | 0.6809 |

---

## 5. Segment the CT scan

Apply the trained model to every picture-band slice and write walk-ready chunks:
```bash
sbatch Code/slurm/seg_05_segment_scan.sh                # z470–1969 -> $WORK/segmentation (75 chunks, ~6 h on one A100)
```
Per batch: CT slices → 20-slice chunks (`ct_slices_to_chunks.py`) → NIfTI (`chunks_to_nifti.py`) →
`nnUNetv2_predict -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 --save_probabilities --disable_tta` →
softmax HDF5 oriented to the CT (`nnunet_softmax_to_h5.py`) → uint8 one-hot (`compact_segmentation.py`). Temporaries (~8 GB per chunk) are deleted after each batch. `Data/3_full_width_ct_scan/segmentation_nnunet3d_picture_band` is exactly this output, so this step can be skipped.

Only the **picture band** is segmented and walked: in the perforation rows the holes break the emulsion for longer than the walk can bridge, and the margins carry no picture emulsion; their geometry is taken from the nearest walked slice at render time.

---

## 6. Scan parameters: seam and picture band

Two constants of this scan are hard-coded in the walk/render scripts. Both were measured from the CT; this job re-derives them (CPU only, not needed to reproduce the results):
```bash
sbatch Code/slurm/param_01_seam_and_picture_band.sh    # -> $WORK/scan_parameters/
```
* **Seam azimuth 353°** (`--seam-deg 353`): the film's outer end terminates at the seam, so the outermost film radius r_out(φ) drops by one winding spacing (~26 px) at a single azimuth on an otherwise smooth curve (re-run: 353.5°, step +22.8 px, 5.4× the noise; the inner-end profile it also prints is dominated by the empty core and is not used) ([`seam_from_outer_boundary.py`](Code/virtual_unrolling/scan_parameters/seam_from_outer_boundary.py)). Generic seam detectors (counting air gaps along rays) are unreliable on this scan, which is why the seam is measured once and passed explicitly.
* **Picture-band limits** ([`picture_band_limits.py`](Code/virtual_unrolling/scan_parameters/picture_band_limits.py)): film coverage of the annulus and the number of air gaps per radius ring (perforation holes) per slice. Full perforation at z ≤ 460 and z ≳ 1990, transitions z 465–600 and z 1935–1985, clean picture z 600–1930 (re-run: identical numbers to the original measurement); the walks cover z470–1949, and frame matching crops the strip rows to the picture (`--z-crop 0.203,0.80`).

---

## 7. Emulsion walk on the segmented data

The emulsion walk ([`virtual_unrolling/emulsion_walk/`](Code/virtual_unrolling/emulsion_walk/)) traces every physical winding as one ordered curve on each segmented slice, links the curves through z, and samples the CT along them:

1. **Seed** — rays from an auxiliary centre count the connected film runs; the 90th-percentile count over rays gives the number of windings; merged runs are split at their interior emulsion runs; one seed per winding on the emulsion centre.
2. **Walk** — predictor–corrector along the emulsion in both directions: step ~2.5 px along the heading, snap perpendicular onto the nearest separate emulsion run, blend the heading; coast at constant radius across short gaps, stop at long ones; stop just short of the spiral seam.
3. **Rescue** — unclaimed emulsion bands between neighbouring walks are seeded and walked.
4. **Link through z** — z-sequential tracking on r(φ) radius profiles (one-to-one assignment, drift compensation, duplicate merge); tracks with low coverage are dropped.
5. **Heal** — per (winding, angle) radius outliers vs the running z-median (> 7.5 px, a jump to a neighbouring winding) are re-interpolated.
6. **Render** — windings are joined inner→outer at the seam; strip columns lie on uniform **arc length** of the film (shared by all z, so rows stay aligned); each pixel averages `--transverse-px` taps across the emulsion thickness.

The walk phase writes a cache (`walk_anchors.npz`); everything after it (tracking, healing, rendering) is re-run from the cache with `--use-cached-anchors`. Always look at the fit-check montage (`walk_anchors_montage.png`: the tracked curves drawn on the CT at several slices) before rendering; the shipped ones are in `Results/2_final_model_full_width/walk_fit_check/`.

```bash
for B in b0 b1 b2 b3 tail; do sbatch Code/slurm/walk_01_walk_range.sh $B; done   # 5 walks in parallel (~1.5-6 h each)
WALKS_FROM=work sbatch Code/slurm/walk_02_merge_and_render.sh                       # merge + render + plain video
sbatch Code/slurm/walk_02_merge_and_render.sh          # same, from the SHIPPED walk caches (reproduces the final strip)
sbatch Code/slurm/walk_02_merge_and_render.sh early    # early version (requested video 1a), from the shipped caches
```
Core commands (`W = python -m virtual_unrolling.emulsion_walk.unroll_whole_roll --ct-dir $D_SCAN/ct_slices`):
```bash
# walk one z-range (b1 shown; every slice, seam forced)
$W --seg-dir <chunks of z770-1069> --out-dir <out> --seam-deg 353.0 \
   --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 --film-min-thick 7.5 --jump-max 5 \
   --snap-accept 7.5 --max-dr 2.5 --recenter-search 7.5 --recenter-sigma 25 \
   --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 --z-heal-px 7.5 \
   --transverse-px 3 --max-infill 3 --radial-robust --seam-exclude-deg 0.3 \
   --n-centers 6 --min-sep 7.5 --min-coverage 100 --inspect-anchors --inspect-max 3
# merge the five caches
python -m virtual_unrolling.emulsion_walk.merge_walk_caches --out <out>/walk_anchors.npz --refit-centers \
   --caches b0.npz b1.npz b2.npz b3.npz tail_dense.npz --z-range - <b0_end+1>: - :1589 1590:1949
# render the final strip from the merged cache
$W --seg-dir $D_SCAN/segmentation_nnunet3d_picture_band --out-dir <out> --use-cached-anchors \
   --seam-deg 353.0 --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
   --z-heal-px 7.5 --min-coverage 300 --transverse-px 4 --seam-exclude-deg 0.3 --seam-bridge
# cut the plain video (no GT)
python -m virtual_unrolling.video.make_film_video <out>/wholeroll.npy <out>/film_final.mp4 \
   --pitch 1130 --invert-phase --reverse --invert --rotate --height 720
```
Expected: 34 windings, strip **(2361, 259297)** (row = z − 40, one column = 1 px of film arc), 229 cells at pitch 1130. From the shipped caches the strip is bit-identical to `Results/2_final_model_full_width/strip/wholeroll_tw4.npy`.

**Early version** (`walk_02_merge_and_render.sh early`): the identical merge and render with `--no-arc-columns --transverse-px 1` (uniform-angle columns, centre-line sampling), cut with the automatic pitch estimate → 34 windings, strip (2361, 259862), 229 cells at pitch 1132.3 px, same framing as the final video. The strip is not shipped (2.4 GB, ~1 h to re-render); the frame sheet and cell boundaries are in `Results/2_final_model_full_width/early_version/`.

---

## 8. Videos and evaluation against the optical scan

```bash
sbatch Code/slurm/eval_01_final_vs_gt.sh       # frame matching, affine, GT-fidelity, GT-free border lock, stillness
```

(a) `frame_match` (`--z-crop 0.203,0.80 --pitch 1130 --no-fliplr`) pairs every GT frame with the film cell re-cut at its best-matching position → `film_stabilized_gtsync.mp4` (video 2b); (b) `stabilize_affine --mode affine` fits one affine warp per frame to the GT → `film_affine.mp4` (video 2a); (c) [`evaluate_final_vs_gt.py`](Code/virtual_unrolling/evaluation/evaluate_final_vs_gt.py) scores it on 227 pairs (last 5 GT frames are a splice/blank tail; GT frame 196 is physically torn), no extra registration, per-pair histogram matching; (d) GT-free border lock (`detect_border` + `border_render`) → video 3b; (e) stillness = temporal-median sharpness of the central 70 % relative to the perforation-locked optical scan (`border_stillness`).

Quick CPU check of the headline numbers (~1 min, no GPU, no SLURM):
```bash
python -m virtual_unrolling.evaluation.evaluate_final_vs_gt \
  --pred Results/0_requested_videos/2a_final_model_WITH_gt_frame_alignment__per_frame_affine.mp4 \
  --gt $GT_VIDEO --out-dir $WORK/check_final [--learned]
```
The perforation-locked reference `Data/4_optical_ground_truth/optical_scan_perflocked_picture.mp4` was made from the optical scan with `python -m virtual_unrolling.evaluation.gt_perflock --gt $GT_VIDEO --out-dir <dir>` (output `gt_perflock_picture.mp4`). The border-lock framing (`--along-shift 0.156` of a pitch) was measured once by registering the temporal median of the render against that of the optical scan.

**Synthetic benchmark + learned unwrapping.** A known video strip is rolled into a CT-like spiral (exact ground-truth mapping), an implicit neural field f(u, z) = analytical spiral + learned residual is fitted to the volume, and the rendered strip is compared with the ground-truth strip and with the *oracle* strip (the CT sampled at the true mapping — the best any mapping can do on this volume):
```bash
sbatch Code/slurm/syn_01_generate.sh                     # both presets (256 x 4096^2, 28 windings) + oracle strips, ~70 GB each
sbatch Code/slurm/syn_02_learned_unwrapping.sh R1S hq_video_4k_outer_imperfect   # supervised on the GT mapping
sbatch Code/slurm/syn_02_learned_unwrapping.sh R1n hq_video_4k_outer_imperfect   # self-supervised
sbatch Code/slurm/syn_02_learned_unwrapping.sh G1  hq_video_4k_outer_imperfect   # supervised, multi-resolution grid encoder
# evaluate a shipped checkpoint only (no training):
CKPT=Results/3_synthetic_learned_unwrapping/R1S_imperfect/train/model_final.pt \
    sbatch Code/slurm/syn_02_learned_unwrapping.sh R1S hq_video_4k_outer_imperfect
```
The shipped checkpoints were trained on an earlier version of the generator that also modulated the emulsion intensity with depth; the geometry is identical and the training losses never read CT intensities, so the learned mapping is the same, and `eval/` scores them on volumes from the current generator. The imperfect preset draws its image noise from numpy's unseeded generator, so each generation is a new noise realisation (scores move in the 4th decimal); the clean preset is deterministic.


---

## 9. Results, and where every reported number comes from

```
Results/
├── 0_requested_videos/                     see §1
├── 1_segmentation/                         fair_comparison_results.json (+ manifest), figures/,
│                                           reproduced_with_deliverable_code/ (seg_04 re-run)
├── 2_final_model_full_width/
│   ├── strip/wholeroll_tw4.npy             THE final strip, float32 (2361, 259297): axis 0 = across the film
│   │                                       (row = z − 40), axis 1 = along the film (1 px of arc per column)
│   ├── scan_parameters/                    seam (353.5°) and picture-band measurements (param_01 output)
│   ├── walk_fit_check/                     traced curves on the CT for each of the five walks
│   ├── plain_render/                       cell boundaries + frame sheet of video 3a
│   ├── early_version/                      cell boundaries + frame sheet of video 1a
│   ├── frame_match/                        GT pairing (matches.json, montage, side_by_side.mp4), videos 2a/2b
│   ├── border_lock/                        detected picture border (border.npz) + video 3b
│   ├── gt_fidelity/                        original_2026-08-20/ (reported scores), reproduced_on_cluster/ (eval_01 job),
│   │                                       reproduced_locally_cpu/{via_mp4,in_memory}/ (same script, laptop CPU)
│   └── stillness/                          thesis_*.json (69 % / 91 % measurements), rerun_on_final_strip/ (71 %)
├── 3_synthetic_learned_unwrapping/         {R1S,R1n}_{clean,imperfect}/, G1_imperfect/: train/ (model_final.pt, train_log,
│                                           geometry.json) + eval/ (eval_metrics.json, strip.npz, figures);
│                                           reproduced_2026-10/ (scores after regenerating the data, VERIFICATION.md)
├── 4_presentation/                         the presentation (PDF)
└── 5_verification_reruns/                  evidence for VERIFICATION.md: comparison logs, verify_compare.py
```

| Reported result | Value | File | Produced by |
|---|---|---|---|
| Segmentation table (Dice …) | 0.9728 … | `Results/1_segmentation/fair_comparison_results.json` | `seg_04_evaluate.sh` |
| GT-fidelity of the final model (227 pairs) | SSIM 0.776, MS-SSIM 0.857, grad-corr 0.788, NMI 1.145, PSNR 21.90 dB, MSAD 0.053, LPIPS 0.143, DISTS 0.125 | `2_*/gt_fidelity/` | `eval_01_final_vs_gt.sh` (`evaluate_final_vs_gt.py`) |
| Stillness vs perforation-locked optical scan | 69 % GT-free, 91 % with GT-estimated along-film offsets (earlier render, §12); 71 % GT-free on the final strip | `2_*/stillness/thesis_*.json`, `2_*/stillness/rerun_on_final_strip/` | `border_render` + `border_stillness` |
| Synthetic, rendered strip vs GT: SSIM | clean: R1S 0.813, R1n 0.809 (oracle 0.977); imperfect: R1S 0.586, R1n 0.385, G1 0.610 (oracle 0.854) | `3_*/<run>/eval/eval_metrics.json` | `syn_01` + `syn_02_learned_unwrapping.sh` |
| Synthetic, row correlation vs GT | clean: R1S 0.9735, R1n 0.9727; imperfect: R1S 0.9726, R1n 0.9030, G1 0.9761 | 〃 | 〃 |

---

## 10. Code map

```
Code/
├── env.sh                          paths (source first)
├── slurm/                          one job script per step (seg_*, param_*, walk_*, eval_*, syn_*) + _activate.sh
├── segmentation/                   standalone scripts:  python Code/segmentation/<script>.py
│   ├── build_dataset503_2d_from_3d.py  Dataset502 -> per-slice Dataset503 (+ its split)
│   ├── nnunet_custom_trainer.py        nnUNetTrainerProgress
│   ├── train_monai.py / predict_monai.py   UNet3D, SwinUNETR
│   ├── fair_compare.py                 evaluation of all five models on identical voxels
│   ├── ct_slices_to_chunks.py / chunks_to_nifti.py / nnunet_softmax_to_h5.py   whole-scan inference I/O
│   └── compact_segmentation.py         float softmax -> verified uint8 one-hot
└── virtual_unrolling/              Python package:  python -m virtual_unrolling.<sub>.<module>
    ├── common/            volume_io.py (CT/segmentation I/O, bilinear sampling), spool_center.py
    ├── scan_parameters/   seam_from_outer_boundary.py, picture_band_limits.py
    ├── emulsion_walk/     walk.py (seeding, predictor-corrector walk, smoothing, re-centring)
    │                      unroll_whole_roll.py (THE pipeline: walk all slices, track, heal, render)
    │                      merge_walk_caches.py, centre_track_report.py
    ├── video/             make_film_video.py (strip -> cells -> mp4)
    ├── evaluation/        frame_match, stabilize_affine, evaluate_final_vs_gt, metrics, video_utils,
    │                      detect_border, border_render, border_stillness, gt_perflock
    ├── synthetic/         generate.py (synthetic CT spiral generator)
    └── learned_unwrapping/  surface_train.py, surface_eval.py, surface_data.py (dataset + analytical
                             spiral base), unwrap_model.py (Fourier-feature MLP), grid_model.py,
                             render_oracle_strip.py
```

The code is the thesis code with clearer names, **compacted**: command-line options, modes and functions that no job script uses were removed (alternative cut/registration/stabilisation modes, legacy presets and patterns, abandoned model types and loss experiments, diagnostics), helpers used by several tools were collected (`common/`, `evaluation/video_utils.py`), and the computations that remain are unchanged. Every job script was re-run with the compacted code and compared with the original code and the shipped results ([`VERIFICATION.md`](VERIFICATION.md)). Deliverable additions are marked in their docstrings (label-only segmentation directories in `discover_volumes`, environment-variable paths in the segmentation scripts, `border_render --no-refine-pitch`). Names in the thesis working repository: `unwrapping/inr/unroll_walk_wholeroll_v13.py` → `emulsion_walk/unroll_whole_roll.py`, `walk_emulsion.py` → `emulsion_walk/walk.py`, `unwrapping/eval/*` → `evaluation/*`, `Scripts/diag_seam_outer.py`, `diag_band_edges.py` → `scan_parameters/*`, `Scripts/diag_centers.py` → `emulsion_walk/centre_track_report.py`, `unwrapping/center_detection.py` → `common/spool_center.py`, `unwrapping/inr/surface_*.py` → `learned_unwrapping/`.

---

## 11. Parameters that matter

These are easy to get wrong and fail **silently** (plausible-looking but wrong output):

* **`--seam-deg 353`** (required) **and `--seam-exclude-deg 0.3`** for every walk/render. A larger exclusion throws away a wedge of film at every winding join; the small value requires the correct seam. A wrong seam shows up as a radial zig-zag of the walk at the assumed seam.
* **Spacing-keyed thresholds** (`--snap-accept 7.5 --max-dr 2.5 --z-heal-px 7.5 --dedup-px 12.5 --track-merge-px 15 …`) are keyed to the 26 px winding spacing of this scan; the argparse defaults are for a 1.25× coarser sampling.
* **Video flags**: `--invert-phase --reverse --invert --rotate` (and `--pitch 1130`, §12). The film is a negative (`--invert`); the strip runs inner→outer, i.e. backwards (`--reverse`); without `--rotate` the picture stands on its side; without `--invert-phase` every cell is cut half a frame off.
* **Mirror.** `frame_match` needs `--no-fliplr`; with its default it matches a mirrored film and still reports a confident (wrong) score (grad-corr 0.106 vs 0.602).
* **Seg orientation.** nnU-Net/NIfTI can silently swap H and W on near-square slices; `nnunet_softmax_to_h5.py` re-orients to the CT and `compact_segmentation.py --ct-dir` refuses mismatched shapes.
* **`--transverse-px 4`** for the final render (the default is 2); 1 = point sample (early version).

---

## 12. Known issues and caveats

* **Segmentation table — "nnU-Net 2D (matched)" is scored on 2, not 5, test volumes.** `build_dataset503_2d_from_3d.py` creates its own `KFold(shuffle=True, random_state=42)` split instead of reusing Dataset502's nnU-Net split. Dataset503's fold-0 training set therefore contains three of Dataset502's five test volumes (1547, 1774, 1847); `fair_compare.py` correctly scores it only on the two volumes it never saw (1263, 1669). The script is shipped unchanged so the reported number remains reproducible; to match the splits, build Dataset503's `splits_final.json` from Dataset502's. Dataset501 (original 2D, independently annotated slices) also contains slices that lie inside test sub-volumes (z1266, 1679, 1784).
* **Stillness measurements.** In `2_*/stillness/thesis_*.json` the 69 % sequence (`CONTROL_nooff`) is the render locked across the film to the printed picture border (GT-free); the 91 % sequence (`FINAL_folded`) additionally folds in along-film translation offsets estimated against the optical scan (per-frame translation registration; that tool is not part of the deliverable). Both were measured on an earlier render of the same walk; `eval_01_final_vs_gt.sh` repeats the GT-free measurement on the final strip (71 %).
* **GT-fidelity script.** The original builder of the matched video pair was not preserved; `evaluate_final_vs_gt.py` re-implements its documented recipe. It reproduces SSIM/MS-SSIM/grad-corr/NMI/PSNR/MSAD to 4 decimals when the pair goes through mp4 (default), and LPIPS/DISTS (0.1430/0.1251) when scored in memory (`--no-via-mp4`); the two paths differ by ≤ 0.0013. `film_affine.mp4` is fitted to the GT per frame, so these scores measure best-aligned content agreement, not raw geometric accuracy.
* **Video pitch.** The automatic pitch estimate on the final strip lands on 1139 px (226 cells, framing creeps over the reel). The 2- and 3-tap renders of the same geometry give 1130 px automatically, so the job scripts pass `--pitch 1130` to `make_film_video` and, for the border-locked video, `detect_border --pitch 1130` + `border_render --no-refine-pitch` (an option added for this deliverable; `border_render` previously always re-estimated the pitch). No GT is involved in either.
* **Physical limits.** The outermost partial turn is walked on too few slices and dropped (229 vs 233 frames); the perforation rows and margins are rendered from clamped picture-band geometry (they look plausible but are not traced — do not use them as fiducials); an unexplained brightness band remains in the left ~15 % of the picture over the first third of the reel.
* **Raw CT is linked, not copied** (§3).

**Verification performed for this deliverable.** Old vs new code on real data: walk cache and rendered strip bit-identical. Cluster re-runs of the scripts in `Code/slurm/` against the original outputs (segmentation table, final strip, GT-aligned videos, GT-fidelity scores, regenerated segmentation, a walk on it): see [`VERIFICATION.md`](VERIFICATION.md) and `Results/5_verification_reruns/`.

---

## 13. Compute requirements

| Step | Resources | Time |
|---|---|---|
| nnU-Net 3D training (fold 0) | 1 × A100-80GB, 16 CPU, 72 GB | ~20 h |
| Segmentation of the picture band | 1 × A100, 180 GB | ~5 min per 20-slice chunk (75 chunks ≈ 6 h) |
| Scan parameters | 4 CPU, 32 GB | ~3 min |
| Walk (one of 5 ranges) | 16 CPU, 180 GB | 1.4 h for b0 (300 slices) |
| Merge + render (2361 × 259297 strip) | 16 CPU, 360 GB | ~1 h |
| Frame matching + affine + evaluation (`eval_01`) | 16 CPU, 360 GB | ~12 min |
| Evaluation of a video vs GT | CPU (GPU for learned metrics) | 1–15 min |
| Synthetic data, one preset + oracle strip | 8 CPU, 160 GB, ~70 GB disk | ~25 min |
| Learned unwrapping: evaluation of one checkpoint | 1 × A100 | ~12 min |

Spool-centre estimation (`find_spool_center`) costs ~10–14 min per call on these slices and dominates fresh walks; `--n-centers` limits the number of calls over the whole stack.
