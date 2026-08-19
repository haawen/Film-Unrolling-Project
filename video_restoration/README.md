# Video restoration / artifact removal — v9 Mickey unroll

Post-processing experiments to clean the v9 unrolled film video, run **after** the
unroll geometry was frozen. Two problem classes, in pipeline order:

1. **De-shading** (broad brightness/banding) — the confirmed dominant artifact.
2. **Denoising + sharpening** (grain/dust) — via temporal consistency (consecutive
   cells share content, so aligned frames can be fused).

Source (input to all of this) = the GT-stabilized de-shaded video
`unwrapping/eval/results/frame_match_v9_geonorm/film_stabilized_gtsync.mp4`.

---

## Methods (each = one script + one results folder)

| # | Method | What it does | Script | Results | Verdict |
|---|--------|--------------|--------|---------|---------|
| 0 | **Geometry-aware de-shading** | Removes the broad brightness field + per-CT-slice banding by dividing out a **content-masked** illumination field (anisotropic: smooth along arc, per-slice along z). Faithful, no halos, no hallucination. | `scripts/01_geo_normalize_deshade.py` (+ prototypes `geo_norm*.py`, `shading_demo.py`, `brightness_stats.py`) | `results/00_deshade/` | ✅ **decent success** (user-approved) |
| 1 | **Classical temporal fusion** | Temporal bilateral filter: per pixel, average ±3 neighbor frames weighted by regional similarity (static bg → denoised, moving char → protected) + unsharp. No training. | `scripts/temporal_fuse.py` | `results/01_classical_fusion/` | ok; sharpens, mild denoise |
| 2 | **FastDVDnet** (pretrained deep) | 5-frame temporal CNN denoiser, zero-shot on our video (σ sweep). Strong grain/dust removal, structure preserved. | `scripts/infer_real.py` + `slurm_fastdvd.sh` | `results/02_fastdvdnet/` | ✅ **strongest so far** (σ≈20) |
| 3 | **UDVD zero-shot** (pretrained blind-spot) | UDVD `blind_video_net` (DAVIS-trained), zero-shot σ sweep. Comparable to FastDVDnet. | `scripts/udvd_infer.py` + `slurm_udvd_zs.sh` | `results/03_udvd_zeroshot/` | ✅ comparable to FastDVDnet |
| 4 | **UDVD-S self-supervised** | Trained UDVD blind-spot on **our own** noisy video (no clean target). | `scripts/udvd_infer.py --blind` + `realvideo_append.py` (dataset) + `patch_train_stable.py` + `slurm_udvd_s*.sh` | `results/04_udvd_s/` | ⚠️ **most conservative** — preserves texture, under-denoises (blind-spot leaves correlated grain) |

**Recommendation:** final chain = **de-shade → FastDVDnet σ20** (→ optional light sharpen).
UDVD-S is the faithful/conservative alternative. See `results/comparisons/compare_all_denoisers.png`.
Not tried: motion-compensated multi-frame SR (BasicVSR++/RVRT) for faithful sharpening; RTN old-film (heavy deps).

`scripts/compare_*.py`, `sxs_*.py`, `compare_strips.py` = montage / side-by-side builders.

## Key comparisons
- `results/comparisons/compare_original_vs_deshaded.mp4` — original vs de-shaded (method 0).
- `results/comparisons/compare_deep.png` — input vs FastDVDnet vs UDVD (methods 2 vs 3).
- `results/02_fastdvdnet/compare_input_vs_fastdvdnet_s20.mp4` — de-shaded vs FastDVDnet.

## Pipeline-critical files that live elsewhere (not moved, other tools depend on them)
- **Cleaned strip:** `unwrapping/inr/results/walk_dense_v9/wholeroll_geonorm.npy` (de-shade output; input to make_film_video / frame_match).
- **De-shaded plain video:** `unwrapping/inr/results/walk_dense_v9/film_v9_geonorm.mp4`.
- **De-shaded stabilized deliverable:** `unwrapping/eval/results/frame_match_v9_geonorm/` (film_stabilized*, side_by_side, pairs).
- De-shade script canonical copy also at `unwrapping/inr/geo_normalize.py`.

## Merlin7 (deep models)
`/data/user/li_k1/M_thesis/video_restore/` — cloned repos (`fastdvdnet/`, `udvd/`, `mf2f/`),
the input video `deshaded_stabilized.mp4`, SLURM scripts, and raw job outputs.
Env: `nnunet` (torch 2.6.0+cu124). GPU jobs → `--clusters=gmerlin7 --partition=a100-hourly --gres=gpu:1`.
