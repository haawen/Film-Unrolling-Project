# Virtual unrolling of a historical film roll

**GitHub contains source code and documentation only.** CT scans, annotations,
trained weights, scores, figures, and videos are distributed separately where
rights permit. Data paths in the commands below describe the layout of that
separate handoff; they are intentionally absent from the GitHub checkout.

This project reconstructs film frames from an X-ray CT scan of a wound reel. The
reproducible path is **CT volume → three-class nnU-Net segmentation → emulsion
walk → flat strip → film cells/video → optional optical-reference evaluation**.
The optical video is used for evaluation and for the explicitly labelled
GT-assisted alignment output; it is not an input to segmentation or walking.

The portable data package is assembled locally in `deliverable/handoff/` and is
not committed to GitHub. Run the commands below from a checkout populated with
the separately supplied handoff data, or from the handoff package root.
Commands use a Bash shell on Linux (the original full-resolution work used
SLURM and A100 80 GB GPUs). The supplied 20-slice walking example can run on a
CPU with sufficient RAM. A full scan run needs substantial RAM, disk, and GPU
time. The 63.5 GiB full-width Mickey CT archive is deliberately **not** in
this handoff; supply your own copy to repeat CT extraction. The supplied
segmentation, walk caches, and final strip support later stages directly.

## Contents and data formats

| Path | Purpose |
|---|---|
| [`Segmentation/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/`](Segmentation/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/) | **Ready to train**: 25 paired `imagesTr/*_0000.nii.gz` and `labelsTr/*.nii.gz`, plus `dataset.json`. One NIfTI case is a 20-slice CT sub-volume. Labels: 0 background, 1 film base, 2 emulsion. The labelled NIfTIs are the files nnU-Net reads for training and validation. |
| [`data/01_Mickey_3d/`](data/01_Mickey_3d/) | **Annotation source exports**: the same 25 CT HDF5 volumes (`volume_XXXX-YYYY.h5`) paired with ilastik three-channel probability exports (`volume_XXXX-YYYY_Probabilities.h5`). These are included in the handoff as well as the training NIfTIs. |
| [`Segmentation/reported_fold0_split.json`](Segmentation/reported_fold0_split.json) | The reported fold: 20 training cases and 5 held-out validation cases. |
| [`Segmentation/nnUNet_data/nnUNet_results/Dataset502_MickeyScroll3D/`](Segmentation/nnUNet_data/nnUNet_results/Dataset502_MickeyScroll3D/) | Trained fold-0 checkpoint, plans, and held-out predictions. Use `checkpoint_final.pth` for the reported model. |
| [`_share/walk_algo_share/sample_data/`](_share/walk_algo_share/sample_data/) | **Ready to walk**: `volume_1595-1614.h5` with key `volume` `(z,y,x)` and paired `*_Probabilities.h5` with key `exported_data` `(z,y,x,3)`. |
| Full-width CT archive (provide separately) | `Mickey_merged_rigid_nobin-tif.zip`: 2,419 indexed TIFF slices, z0000–2418. It is excluded to avoid transferring 63.5 GiB already available to the recipient. This is a **different acquisition of the same film** from the older 233-cell picture-area scan. |
| [`verified_reference/`](verified_reference/README.md) | Independently rerun deliverable with its own compacted Python package, exact job scripts, verification logs, five cached walks, 75 compact full-width segmentation chunks, expanded video set, matched 2D checkpoint, per-pair learned scores, and synthetic benchmark artifacts. The Dataset501/502 training data and final strip are stored once at the parent handoff paths shown above and below; the verified scripts point there. Its `Data/3_full_width_ct_scan/ct_slices/` must be supplied from the separate CT archive for CT-dependent reruns. |
| [`data/h265_1080p.mp4`](data/h265_1080p.mp4) | Optical reference used only for frame matching, alignment, and scoring. |
| [`Segmentation/fair_comparison/fair_comparison_results.json`](Segmentation/fair_comparison/fair_comparison_results.json) and [`unwrapping/eval/results/affine_method_comparison/scores.json`](unwrapping/eval/results/affine_method_comparison/scores.json) | Per-case segmentation and per-variant reconstruction measurements. |

### Hand-labelled segmentation data and ilastik provenance

The thesis draft describes the annotation process as semi-automatic: an ilastik
random-forest pixel classifier produced per-slice class probability maps; these
provided initial labels, which were reviewed and corrected slice by slice in
3D Slicer. The 25 exported CT/probability pairs are in
[`data/01_Mickey_3d/`](data/01_Mickey_3d/). Each CT file has HDF5 key `volume`
with shape `(20, 3063, 3062)`; its companion has key `exported_data` with shape
`(20, 3063, 3062, 3)`. Channels 0, 1, and 2 correspond to background, film
base, and emulsion. The labelled training masks are the paired
[`labelsTr/*.nii.gz`](Segmentation/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/labelsTr/)
files; the CT inputs are under
[`imagesTr/`](Segmentation/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTr/).

[`Scripts/convert_3d_data.py`](Scripts/convert_3d_data.py) converts the exported
probability maps to categorical NIfTI masks by `argmax` over the three channels.
The available files do not include the ilastik `.ilp` project, the original
brush strokes, or a separate 3D Slicer correction history. Thus the saved
labels can be used to reproduce training, but the historical editing session
cannot be replayed. In one checked case (`Mickey3D_0836`, slices 0, 10, 19),
the saved masks exactly match the exported probability `argmax`; the extent to
which Slicer corrections were incorporated into the exported maps is not
independently recorded.

The supplied Dataset502 directory is already in nnU-Net's required layout:
`imagesTr/Mickey3D_XXXX_0000.nii.gz` pairs with
`labelsTr/Mickey3D_XXXX.nii.gz`, and `dataset.json` declares the three classes.
Set `nnUNet_raw` as shown below and run Section 1; there is no import step.
To regenerate NIfTIs from the HDF5 exports in a **separate** directory for
inspection, run:

```bash
python Scripts/convert_3d_data.py \
  --hdf5-dir data/01_Mickey_3d --nnunet-raw work/rebuilt_nnUNet_raw
# To train on that rebuilt copy, set this after the Install block below and
# immediately before the planning/preprocessing commands in Section 1:
export nnUNet_raw="$(pwd)/work/rebuilt_nnUNet_raw"
```

The full raw CT for the **older picture-area scan** is still absent beyond
these 25 annotation volumes; consequently, its 233-cell ablation cannot be
recomputed from CT. Its derived strip, video, and measurement JSON are included
in the handoff for output and evaluation inspection.

## Install

Use Python 3.12 and a CUDA-compatible PyTorch installation for nnU-Net. Install
PyTorch for the CUDA version on the target machine first, then install the
project's [`requirements.txt`](requirements.txt). The requirements cover
nnU-Net, HDF5/NIfTI/TIFF I/O, walking, video rendering, and evaluation.

```bash
conda create -n film-unroll python=3.12 -y
conda activate film-unroll
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The independently verified reruns under `verified_reference/` used their own
[`requirements.txt`](verified_reference/requirements.txt), including the
recorded `pyiqa==0.1.15.post2` version. Use that file in a separate environment
when matching its learned image-quality scores exactly; the root requirements
serve the original workspace modules and use `pyiqa==0.1.16`.

The video scripts use `imageio-ffmpeg`, installed through `requirements.txt`.
`pyiqa` is needed for learned LPIPS scores; basic Dice and geometric
metrics do not require it. The handoff includes the AlexNet and LPIPS v0.1
weights used by the ablation evaluator. Set `TORCH_HOME` to that bundled
cache, then set the project paths and make the repository's
250-epoch trainer discoverable to nnU-Net:

```bash
export PROJECT_ROOT="$(pwd)"
if [ -d "$PROJECT_ROOT/weights/torch" ]; then
  export TORCH_HOME="$PROJECT_ROOT/weights/torch"
else
  export TORCH_HOME="$PROJECT_ROOT/deliverable/weights/torch"
fi
export nnUNet_raw="$PROJECT_ROOT/Segmentation/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/Segmentation/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/Segmentation/nnUNet_data/nnUNet_results"
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
python - <<'PY'
from pathlib import Path
import shutil
import nnunetv2
root = Path.cwd()
target = Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/custom_trainer.py'
shutil.copy2(root / 'Scripts/custom_trainer.py', target)
print(target)
PY
```

### Python entry points for the pipeline

The commands in the following sections call these project Python files. The
`nnUNetv2_*` commands themselves come from the installed `nnunetv2` package;
the project's selected trainer is [`Scripts/custom_trainer.py`](Scripts/custom_trainer.py).
[`Scripts/reproduce_fullwidth_batch.sh`](Scripts/reproduce_fullwidth_batch.sh)
automates the full-width chunk loop; the Python files below contain the
individual processing stages it invokes.

| Step | Main Python source |
|---|---|
| Rebuild training NIfTIs from ilastik HDF5 exports, if needed | [`Scripts/convert_3d_data.py`](Scripts/convert_3d_data.py) |
| Train the selected 3D segmentation model | [`Scripts/custom_trainer.py`](Scripts/custom_trainer.py), loaded by `nnUNetv2_train` |
| Score held-out segmentation masks | [`Scripts/evaluate_segmentation.py`](Scripts/evaluate_segmentation.py) |
| Extract full-width CT TIFFs to HDF5 | [`Scripts/mickey_zip_export_h5.py`](Scripts/mickey_zip_export_h5.py) |
| Build 20-slice CT chunks and nnU-Net inference inputs | [`Scripts/newscan_seg_test.py`](Scripts/newscan_seg_test.py), [`Scripts/hdf5_to_nifti_for_predict.py`](Scripts/hdf5_to_nifti_for_predict.py) |
| Convert predicted probabilities back to walking input | [`Scripts/nnunet_softmax_to_h5_probs.py`](Scripts/nnunet_softmax_to_h5_probs.py); inference itself uses `nnUNetv2_predict` |
| Walk one real 20-slice sample | [`unwrapping/inr/walk_emulsion.py`](unwrapping/inr/walk_emulsion.py) |
| Walk the full-width scan and render its flat strip | [`unwrapping/inr/unroll_walk_wholeroll_v13.py`](unwrapping/inr/unroll_walk_wholeroll_v13.py); batch cache merger: [`Scripts/merge_walk_caches.py`](Scripts/merge_walk_caches.py) |
| Make an initial video from the flat strip | [`unwrapping/inr/make_film_video.py`](unwrapping/inr/make_film_video.py) |
| Produce the canonical GT-free, printed-border-aligned video | [`verified_reference/Code/virtual_unrolling/evaluation/detect_border.py`](verified_reference/Code/virtual_unrolling/evaluation/detect_border.py), then [`verified_reference/Code/virtual_unrolling/evaluation/border_render.py`](verified_reference/Code/virtual_unrolling/evaluation/border_render.py) |
| Prepare the optical GT alignment reference | [`unwrapping/eval/gt_perflock.py`](unwrapping/eval/gt_perflock.py) |
| Produce the canonical scored GT-assisted video | [`verified_reference/Code/virtual_unrolling/evaluation/frame_match.py`](verified_reference/Code/virtual_unrolling/evaluation/frame_match.py), then [`verified_reference/Code/virtual_unrolling/evaluation/stabilize_affine.py`](verified_reference/Code/virtual_unrolling/evaluation/stabilize_affine.py) |
| Score the ablation videos against optical GT | [`unwrapping/eval/compare_affine_variants.py`](unwrapping/eval/compare_affine_variants.py) |

## 1. Train the selected 3D U-Net segmentation model

The reported model is **nnU-Net v2 Dataset502, `3d_fullres`, fold 0,
`nnUNetTrainerProgress` (250 epochs)**. The prepared dataset is already under
`$nnUNet_raw`; no data conversion is needed. The custom split ensures the five
held-out cases are the same as in the reported comparison.

```bash
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres
cp Segmentation/reported_fold0_split.json \
  "$nnUNet_preprocessed/Dataset502_MickeyScroll3D/splits_final.json"
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerProgress --npz
```

Training writes `checkpoint_final.pth` under
`$nnUNet_results/Dataset502_MickeyScroll3D/nnUNetTrainerProgress__nnUNetPlans__3d_fullres/fold_0/`.
The provided checkpoint allows inference without retraining. The custom
trainer's 250-epoch schedule is in [`Scripts/custom_trainer.py`](Scripts/custom_trainer.py).
Other architectures and the 2D comparison are in
[`Scripts/slurm/`](Scripts/slurm/) and [`Scripts/train_monai.py`](Scripts/train_monai.py);
those experiments are separate from the selected model used for walking.

## 2. Evaluate segmentation on held-out reference labels

Run validation after training, or use the five validation NIfTIs supplied with
the checkpoint. `--val` uses the fold-0 checkpoint and writes predictions in
`fold_0/validation/`. The scorer reproduces the comparison's macro mean over
**film base and emulsion**, for each of the five held-out cases; background is
excluded from this reported number.

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerProgress --val --npz
python Scripts/evaluate_segmentation.py \
  --pred-dir "$nnUNet_results/Dataset502_MickeyScroll3D/nnUNetTrainerProgress__nnUNetPlans__3d_fullres/fold_0/validation" \
  --labels-dir "$nnUNet_raw/Dataset502_MickeyScroll3D/labelsTr" \
  --output work/segmentation_validation.json
```

The shipped prediction and label pairs yield **Dice 0.972804** and **IoU
0.947521**; the independently recomputed result is in
[`segmentation_validation_recomputed.json`](Segmentation/segmentation_validation_recomputed.json).
The other reported architectures
score 0.971677 Dice (matched 2D nnU-Net), 0.942534 (MONAI UNet3D), 0.783940
(SwinUNETR), and 0.748472 (original 2D nnU-Net). The earlier README's
SwinUNETR value was inconsistent with the recorded results and has been
corrected here. The machine-readable result is in the data table above.

## 3. Segment CT data and run the Walking algorithm

### Small, directly runnable real-data example

The included HDF5 pair is already segmented with three-class probabilities.
The `walk_emulsion` CLI takes argmax internally and creates CT/segmentation
overlays plus an unrolled strip preview. This is the fastest end-to-end check
of the walking module and its data format:

```bash
python -m unwrapping.inr.walk_emulsion \
  --vol _share/walk_algo_share/sample_data/volume_1595-1614.h5 \
  --probs _share/walk_algo_share/sample_data/volume_1595-1614_Probabilities.h5 \
  --out-dir work/walk_sample
```

Expected output includes `walk_ct.png`, `walk_seg.png`, `walk_zoom.png`,
`strip_overview.png`, and `strip_crop.png`. This sample's documented run found
32 deduplicated windings and a `(20, 65024)` strip. See
[`_share/walk_algo_share/README.md`](_share/walk_algo_share/README.md) for the
single-chunk algorithm explanation and all tuning flags (`--help`).

### Inference on the full-width scan

If starting from CT, provide the excluded full-width scan archive separately
and convert it to per-slice HDF5. The same scan's **75 segmented picture-band
chunks** and **five walk caches** are already under
[`verified_reference/Data/3_full_width_ct_scan/`](verified_reference/Data/3_full_width_ct_scan/),
so segmentation and walking can also be inspected without rerunning them. Its
picture band is approximately z470–1969; the perforation bands are outside
this range. The full-width run walks the picture band and extends the nearest
geometry into the perforation bands. Never mix its 3738×3769 slices with the
older 3063×3062 picture-area scan.

```bash
CT_ZIP=/path/to/Mickey_merged_rigid_nobin-tif.zip
python Scripts/mickey_zip_export_h5.py \
  --zip "$CT_ZIP" \
  --z-ranges 40-2400 --stage work/01_Mickey_sprockets

# Example first 20-slice picture-band chunk; repeat at z=490, 510, ... 1950.
python Scripts/newscan_seg_test.py chunks \
  --in-dir work/01_Mickey_sprockets --out-dir work/pairs \
  --z0 470 --z1 489 --scale 1.0
python Scripts/hdf5_to_nifti_for_predict.py \
  --in-dir work/pairs --out-dir work/nii --prefix NewScan
nnUNetv2_predict -i work/nii -o work/pred \
  -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
  --save_probabilities --disable_tta
python Scripts/nnunet_softmax_to_h5_probs.py \
  --nnunet-out work/pred --pairs-dir work/pairs --prefix NewScan
```

The final command writes `volume_0470-0489_Probabilities.h5` beside the CT
chunk, with key `exported_data` and class order 0/1/2. For the full run, use
the portable batch wrapper below. It applies the same segmentation and walk
parameters to fifteen chunks at a time, saves one `walk_anchors.npz` per
batch, and removes the large temporary probability arrays only after that
cache exists. Run the five batches sequentially or as independent cluster jobs.

```bash
bash Scripts/reproduce_fullwidth_batch.sh 0
bash Scripts/reproduce_fullwidth_batch.sh 1
bash Scripts/reproduce_fullwidth_batch.sh 2
bash Scripts/reproduce_fullwidth_batch.sh 3
bash Scripts/reproduce_fullwidth_batch.sh 4

mkdir -p work/fullwidth/full
python Scripts/merge_walk_caches.py \
  --out work/fullwidth/full/walk_anchors.npz \
  --caches work/fullwidth/b0/walk/walk_anchors.npz \
           work/fullwidth/b1/walk/walk_anchors.npz \
           work/fullwidth/b2/walk/walk_anchors.npz \
           work/fullwidth/b3/walk/walk_anchors.npz \
           work/fullwidth/b4/walk/walk_anchors.npz

python -m unwrapping.inr.unroll_walk_wholeroll_v13 \
  --ct-dir work/01_Mickey_sprockets --seg-dir work/fullwidth \
  --out-dir work/fullwidth/full --use-cached-anchors \
  --seam-deg 353.0 --ref-mode track --min-sep 13 \
  --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
  --z-heal-px 7.5 --min-coverage 400 --transverse-px 3 \
  --seam-exclude-deg 0.3 --seam-bridge
python -m unwrapping.inr.make_film_video \
  work/fullwidth/full/wholeroll.npy work/fullwidth/full/film.mp4 \
  --phase-lock global --invert-phase --reverse --invert --rotate --height 720
```

The `--seg-dir` argument in the cached render is required by the CLI but not
read when `--use-cached-anchors` is set. The 400-anchor coverage threshold
assumes all five batches completed. The initial `film.mp4` is GT-free. The two
final alignment modes below start from the rendered `wholeroll.npy` strip.

### Generate the final GT-free aligned video

[`unwrapping/eval/detect_border.py`](unwrapping/eval/detect_border.py) finds
the film's printed picture border in the CT-derived strip, and
[`unwrapping/eval/border_render.py`](unwrapping/eval/border_render.py) aligns
frames to that border. **Neither command reads the optical GT video.** This
produces the original workspace's alternate `film_borderlock_m02.mp4` variant.
The canonical top-level handoff video is regenerated below using the verified
reference code:

```bash
STRIP=work/fullwidth/full/wholeroll.npy
BORDER=work/fullwidth/border/border.npz
OUT=work/fullwidth/aligned
python -m unwrapping.eval.detect_border "$STRIP" --out-dir work/fullwidth/border
python -m unwrapping.eval.border_render "$STRIP" \
  --border "$BORDER" --out-dir "$OUT" --along phaselock \
  --smooth-frames 0.8 --reverse --invert --height 720 \
  --along-shift 0.156 --across-shift 0.0 --across-scale 1.0 \
  --margin 0.02 --video film_borderlock_m02.mp4
```

### Generate the final GT-assisted aligned video

The optical reference is the included
[`gt_perflock_picture.mp4`](unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4).
To rebuild that reference from [`data/h265_1080p.mp4`](data/h265_1080p.mp4),
run [`unwrapping/eval/gt_perflock.py`](unwrapping/eval/gt_perflock.py):

```bash
python -m unwrapping.eval.gt_perflock \
  --gt data/h265_1080p.mp4 --out-dir work/fullwidth/gt
```

Then use `work/fullwidth/gt/gt_perflock_picture.mp4` as `GT` below.

[`unwrapping/eval/stabilize_horizontal.py`](unwrapping/eval/stabilize_horizontal.py)
measures GT-derived along-film offsets from the **GT-free output above**.
[`unwrapping/eval/border_render.py`](unwrapping/eval/border_render.py) applies
those offsets during rendering, and
[`unwrapping/eval/stabilize_affine.py`](unwrapping/eval/stabilize_affine.py)
performs the final per-frame affine registration against GT. This produces the
original workspace's alternate `FLAGSHIP_borderlock_affine.mp4` variant. The
canonical top-level handoff video follows the verified reference path below:

```bash
STRIP=work/fullwidth/full/wholeroll.npy
BORDER=work/fullwidth/border/border.npz
OUT=work/fullwidth/aligned
GT=unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4
python -m unwrapping.eval.stabilize_horizontal \
  --in "$OUT/film_borderlock_m02.mp4" --out "$OUT/scratch_offsets.mp4" \
  --gt "$GT" --gt-crop 0,0,1,1 --axes v \
  --save-offsets "$OUT/along_offsets.npy"
python -m unwrapping.eval.border_render "$STRIP" \
  --border "$BORDER" --out-dir "$OUT" --along phaselock \
  --smooth-frames 0.8 --reverse --invert --height 720 \
  --along-shift 0.156 --across-shift 0.0 --across-scale 1.0 \
  --along-offsets "$OUT/along_offsets.npy" --along-offsets-sign +1 \
  --margin 0.02 --video FLAGSHIP_borderlock.mp4
python -m unwrapping.eval.stabilize_affine \
  --in "$OUT/FLAGSHIP_borderlock.mp4" --gt "$GT" \
  --gt-crop 0,0,1,1 --mode affine \
  --out "$OUT/FLAGSHIP_borderlock_affine.mp4"
```

The alternate commands above generalize the original PSI-specific
[`batch`](Scripts/slurm/slurm_newscan_dense_batch.sh),
[`render`](Scripts/slurm/slurm_newscan_dense_render_ms13.sh), and
[`alignment`](Scripts/slurm/slurm_newscan_flagship.sh) jobs.

### Rebuild the canonical final videos from the supplied strip

The two top-level `videos/02_*` and `videos/03_*` files now match the
independently verified results in [`verified_reference/`](verified_reference/README.md).
Their source strip is
[`wholeroll_tw4.npy`](unwrapping/inr/results/newscan/render/wholeroll_tw4.npy),
stored once under `unwrapping/inr/results/newscan/render/`.
GT-assisted matching chooses each film-cell cut from the optical reference,
then applies a per-frame affine warp. GT-free rendering tracks the printed
border in the CT-derived strip and does not read the optical video. From the
handoff root, run:

```bash
cd verified_reference
source Code/env.sh
STRIP=../unwrapping/inr/results/newscan/render/wholeroll_tw4.npy
OUT=work/rebuilt_videos
python -m virtual_unrolling.evaluation.frame_match \
  --strip "$STRIP" --gt "$GT_VIDEO" --z-crop 0.203,0.80 --pitch 1130 \
  --no-fliplr --export-video --montage --video --fps 25 \
  --out-dir "$OUT/frame_match"
python -m virtual_unrolling.evaluation.stabilize_affine \
  --in "$OUT/frame_match/film_stabilized_gtsync.mp4" --gt "$GT_VIDEO" \
  --out "$OUT/frame_match/film_affine.mp4"

python -m virtual_unrolling.evaluation.detect_border \
  "$STRIP" --out-dir "$OUT/border" --pitch 1130
python -m virtual_unrolling.evaluation.border_render \
  "$STRIP" --border "$OUT/border/border.npz" \
  --out-dir "$OUT/border_lock" --smooth-frames 0.8 \
  --reverse --invert --height 720 --along-shift 0.156 \
  --margin 0.02 --no-refine-pitch --video film_borderlock_noGT.mp4
```

The outputs `film_affine.mp4` and `film_borderlock_noGT.mp4` correspond to the
canonical top-level video 02 and video 03. The verified package's
[`eval_01_final_vs_gt.sh`](verified_reference/Code/slurm/eval_01_final_vs_gt.sh)
also runs these steps plus quantitative scoring and stillness checks; its
[`VERIFICATION.md`](verified_reference/VERIFICATION.md) records the reruns.
To regenerate the final strip or early full-width video from the five shipped
walk caches, first supply the excluded CT as per-slice HDF5 under
`verified_reference/Data/3_full_width_ct_scan/ct_slices/`, then follow
[`walk_02_merge_and_render.sh`](verified_reference/Code/slurm/walk_02_merge_and_render.sh).

## 4. Evaluate the reconstructed frames

The four-row ablation in the presentation uses a fixed frame-matching, crop,
and affine alignment protocol. The recorded source is
[`scores.json`](unwrapping/eval/results/affine_method_comparison/scores.json):

| Variant | Source video | Grad. corr. | MS-SSIM | LPIPS |
|---|---|---:|---:|---:|
| Global curve baseline | `film_final_p902.mp4` | 0.308 | 0.414 | 0.474 |
| Emulsion walk, before arc-length correction | `film_v9.mp4` | 0.373 | 0.543 | 0.317 |
| Arc-length columns | `film_v13.mp4` | 0.589 | 0.685 | 0.242 |
| Three-sample transverse average | `film_v13_tw3.mp4` | 0.595 | 0.685 | 0.241 |

To recompute the four ablation rows from the supplied videos with the recorded
common affine protocol, run:

```bash
python -m unwrapping.eval.compare_affine_variants \
  --gt data/h265_1080p.mp4 --out work/ablation_scores.json --lpips \
  --variant 'Polynomial baseline=unwrapping/inr/results/arc_simple/wholeroll_v2/film_final_p902.mp4' \
  --variant 'Emulsion walk=unwrapping/inr/results/walk_dense_v9/film_v9.mp4' \
  --variant 'Arc-length sampling=unwrapping/inr/results/walk_dense_v13/film_v13.mp4' \
  --variant 'Transverse average=unwrapping/inr/results/walk_dense_v13_tw3/film_v13_tw3.mp4'
```

The protocol settings and per-frame scores are also in the recorded
`affine_method_comparison/scores.json`; changing crop, matching, or alignment
changes the values. The separate full-width clean-pair score file
is [`scores_clean.json`](unwrapping/eval/results/newscan_affine_vs_gt_trim/scores_clean.json):
227 retained pairs after excluding a damaged optical frame, gradient
correlation 0.787930, MS-SSIM 0.857223, SSIM 0.775736, PSNR 21.9029 dB.
The presentation's full-width LPIPS 0.143 is now supported by the verified
package's [`per-pair scores`](verified_reference/Results/2_final_model_full_width/gt_fidelity/reproduced_locally_cpu/in_memory/scores.json):
the 227 retained pairs have mean LPIPS 0.143036. Its
[`evaluate_final_vs_gt.py`](verified_reference/Code/virtual_unrolling/evaluation/evaluate_final_vs_gt.py)
and [`VERIFICATION.md`](verified_reference/VERIFICATION.md) document the
measurement and rerun protocol.

## Requested videos and handoff status

The requested MP4s are in [`deliverable/handoff/videos/`](deliverable/handoff/videos/):

1. `01_emulsion_walk_ablation.mp4` — the exact `film_v9.mp4` used for the
   presentation's “Emulsion walk” ablation row.
2. `01a_emulsion_walk_full_width.mp4` — the early walk on the final full-width
   scan before arc-length columns and transverse averaging; its walk caches
   are supplied, so it can be regenerated if the recipient provides CT.
3. `02_final_gt_assisted_alignment.mp4` — the verified final model with
   GT-chosen frame cuts and per-frame affine alignment; this is the video
   evaluated for the reported full-width image-quality scores.
4. `03_final_gt_free_alignment.mp4` — the verified final model border-locked
   to its own printed border, without optical GT alignment.

The MP4s contain 232, 229, 233, and 228 encoded frames respectively. The two
full-width alignment outputs use different endpoint cuts and playback rates
(26 versus 25 fps); frame rate is a viewing setting, not the film's historical
capture rate. Match cells before comparing them frame by frame. The earlier
GT-anchored-borderlock and borderlock viewing variants remain under
`unwrapping/inr/results/newscan/stabilized/` for comparison.

The handoff includes the ready-to-use training NIfTIs, fold split, checkpoint,
the 25 paired CT/ilastik HDF5 exports, real 20-slice walking pair, derived
full-width segmentation, cached walks, final strip, optical reference, matched
2D checkpoint, synthetic benchmark results, scripts, score JSONs, and videos.
The full-width CT archive is excluded by request. The package's
[`MANIFEST.json`](deliverable/handoff/MANIFEST.json) lists every file, its size,
and SHA-256 hashes for the named videos and bundled LPIPS weights. The
main remaining gaps are:

- The ilastik `.ilp` project and separate 3D Slicer edit history are absent;
  the preserved probability exports and final NIfTI labels support training,
  but the annotation session cannot be reconstructed exactly.
- The full CT and dense segmentation for the **older** 233-cell ablation scan
  were not found. Its provided video/strip and score JSON support visual and
  video-level evaluation, but not a fresh CT-to-video reconstruction.
- The 63.5 GiB full-width CT archive is omitted as requested. The recipient
  needs a separate copy to repeat CT extraction or render from cached walks;
  the full-width segmented chunks, caches, strip, videos, and scores are here.
- The original workspace's alternate `newscan/render/wholeroll.npy` and its
  GT-derived along-film offsets are absent. The canonical verified alignment
  path above uses the supplied `wholeroll_tw4.npy`, matches, and border track.

A Git checkout alone will not contain the large data files, because
`.gitignore` excludes them; transfer the handoff directory with its data files.
