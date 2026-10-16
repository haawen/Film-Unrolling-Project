# Verification of this deliverable

Before hand-over, every pipeline step was re-run **with the code and job scripts
shipped in `Code/`** (on PSI Merlin7, October 2026) and compared with the outputs
that were reported. "Bit-identical" means every value of the array / every byte of
the file is the same; "pixel-identical" means every decoded video frame is the same
(the mp4 container bytes may differ between encoder builds). The raw comparison logs
and the comparison script are in
[`Results/5_verification_reruns/`](Results/5_verification_reruns/).

## 1. Reproduction of the reported results

| Step (script) | Compared with | Result |
|---|---|---|
| Segmentation evaluation (`seg_04_evaluate.sh`, shipped models) | reported segmentation table | aggregate Dice/IoU/precision/recall of all 5 models **identical to 6 decimals** (largest difference 5 × 10⁻⁷; the evaluation re-runs every model's GPU inference, whose run-to-run noise is of that size) (`Results/1_segmentation/reproduced_with_deliverable_code/`) |
| Final render (`walk_02_merge_and_render.sh`, shipped walk caches) | original strip (2361 × 259297) | **bit-identical** (34 windings, 417 seam-bridge columns); the plain video is pixel-identical to `3a_final_model_NO_gt__plain_render.mp4` |
| Frame matching + affine (`eval_01_final_vs_gt.sh`) | original `film_stabilized_gtsync.mp4`, `film_affine.mp4` | **byte-identical** (md5 e4e39fce…, 26040ad3…) |
| GT-fidelity (`evaluate_final_vs_gt.py`, in `eval_01`) | reported 227-pair scores | grad-corr 0.7879, MS-SSIM 0.8572, SSIM 0.7758, NMI 1.1449, PSNR 21.903 dB, MSAD 0.0526 (identical to 4 decimals); LPIPS 0.1429, DISTS 0.1257 (reported 0.143 / 0.125; the in-memory path `--no-via-mp4` gives 0.1430 / 0.1251) |
| GT-free border lock + stillness (`eval_01`) | — | border-locked video = `3b_final_model_NO_gt__border_locked.mp4`; stillness 0.360 vs 0.508 for the perforation-locked optical scan = **71 %** |
| Scan parameters (`param_01_seam_and_picture_band.sh`, CT only) | recorded seam and band limits | seam **353.5°** (outer-boundary step +22.8 px, 5.4× the noise); band statistics equal to the original measurement (e.g. z470: coverage 0.669, 11.13 gaps/ring; z500: 0.831, 6.08) |
| Segmentation regenerated (`seg_05_segment_scan.sh 470 1589`) | surviving original nnU-Net output (z530–779) | voxel label agreement **99.33 %** over 3.5 × 10⁹ voxels; Dice(original, regenerated) film base 0.991, emulsion 0.972. Not identical by construction: that surviving original was cut into 20-slice chunks starting at z520, the regenerated one at z470, so nnU-Net sees a different z-context for every slice |
| Walk b0 from scratch on the **regenerated** segmentation (`walk_01_walk_range.sh b0`, 300 anchors, 1.4 h) | shipped `walk_caches/b0.npz` (the original walk) | same anchors, same spool centres (0.00 px), same number of walks (11 354); per walk, median radial difference 0.010 px; 99.89 % of walks within 0.5 px |
| Synthetic benchmark regenerated (`syn_01` commands, imperfect preset, 256 × 4096²) + oracle strip + re-evaluation of the shipped checkpoints | reported flat-slab scores | R1S SSIM 0.58589 vs 0.58591, R1n 0.38527 vs 0.38471, G1 0.61071 vs 0.61028, oracle 0.85424 vs 0.85422 (row correlation, PSNR likewise within 10⁻³). Not bit-identical because the imperfect preset draws its image noise (σ = 0.03) from numpy's unseeded global generator, so every generation is a new noise realisation; the clean preset is deterministic and reproduces **exactly**: R1S 0.812812, R1n 0.808567, oracle 0.977453, row correlation 0.97347 / 0.97273 (`Results/3_synthetic_learned_unwrapping/reproduced_2026-10/`) |
| Early version (`walk_02_merge_and_render.sh early`, shipped walk caches) | — (new) | 34 windings, strip (2361, 259862), 229 cells at an automatic pitch of 1132.3 px; same framing as the final video |

## 2. Code compaction

The code was compacted after the runs above: options, modes and functions that no
job script uses were removed (~48 % fewer lines, 42 instead of 68 Python files)
and the remaining computations are unchanged. Every job script was then run again with the
compacted code, and its outputs compared with the same job run on the original code
("baseline") or with the shipped results:

| Re-run with compacted code | Compared with | Result |
|---|---|---|
| `walk_01_walk_range.sh b1` and `tail` (one chunk each) | baseline | walk caches **identical** (768 and 753 walks) |
| `walk_02_merge_and_render.sh` | shipped strip, video 3a | strip **bit-identical**, video **pixel-identical** |
| `walk_02_merge_and_render.sh early` | baseline strip, video 1a | strip **bit-identical**, video **pixel-identical** |
| `eval_01_final_vs_gt.sh` | shipped videos, baseline scores | GT-aligned videos **byte-identical**, border-locked video **pixel-identical** to video 3b, GT-fidelity and stillness scores **identical** |
| `param_01_seam_and_picture_band.sh` | baseline | logs identical (apart from file paths) |
| `gt_perflock` | baseline | perforation-locked GT video **pixel-identical** |
| Synthetic generator + oracle strip (both presets, 6 slices; same numpy seed) | baseline | all 42 arrays **identical** |
| `seg_05_segment_scan.sh` (one chunk), `seg_04_evaluate.sh` | baseline | segmentation chunk **identical**; evaluation scores equal within GPU run-to-run noise (aggregate ≤ 3 × 10⁻⁷; `fair_compare.py` itself is unchanged) |
| `surface_eval` on the shipped R1S / R1n / G1 checkpoints | baseline | metrics and rendered strips **identical** (on the reduced 6-slice set) |
| `surface_train` R1S / R1n / G1, 40 steps, same seed | baseline (and baseline vs baseline) | final weights and every logged loss **identical**; two baseline runs are identical too, so the comparison is exact |

Import check: all 32 package modules import; every intra-package import resolves.
