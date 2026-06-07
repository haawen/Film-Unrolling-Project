#!/bin/bash
# =============================================================================
# Round 1 — attachment target × supervision matrix, run as a single variant.
#
#   Variants:
#     R1a  emulsion band    + pure self-supervised
#     R1b  emulsion band    + supervised warm-start → SS release
#     R1c  emulsion centerline (erode=1) + pure self-supervised
#     R1d  emulsion centerline (erode=1) + supervised warm-start → SS release
#
# Usage (smoke):
#     sbatch Scripts/slurm/slurm_R1_variant.sh R1a smoke
#     sbatch Scripts/slurm/slurm_R1_variant.sh R1b smoke
#     sbatch Scripts/slurm/slurm_R1_variant.sh R1c smoke
#     sbatch Scripts/slurm/slurm_R1_variant.sh R1d smoke
#
# Usage (full):
#     sbatch Scripts/slurm/slurm_R1_variant.sh R1a full
#     ... etc.
#
# Smoke = all 20 synthetic slices, 1500 steps, a100-hourly, synthetic-only eval.
# Full  = all 20 synthetic slices, 10000 steps, a100-daily, synthetic eval
#         + additional eval on real Mickey (qualitative transfer check).
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=R1
#SBATCH --output=logs/R1_%x_%j.out
#SBATCH --error=logs/R1_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_R1_variant.sh <variant> <smoke|full> [preset]}"
MODE="${2:?usage: sbatch slurm_R1_variant.sh <variant> <smoke|full> [preset]}"
PRESET="${3:-clean_4k}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
REAL_DIR="${PROJECT_DIR}/01_Mickey_3d"

# Default LR (overridable per variant). Used to test the "LR-at-handoff"
# hypothesis: smoke worked because cosine-LR put it at ~25% of peak by SS
# release, full failed because it was at ~79%.
LR="1e-3"
SUP_OVERRIDE=""   # if set, replaces the smoke/full step budget shaping
LEARN_ECC=0       # R2b: learnable analytical eccentricity (amp·cos(theta-phase))
HIDDEN_DIM=256    # R2c: INR hidden dim (default matches surface_train.py)
N_LAYERS_MLP=4    # R2c: INR depth
W_ZCOH=0.0        # R3: z-coherence weight (TV on pred_xy across z)
PW_ECC=0          # R3 lever 2: per-winding (amp_k, phase_k) ecc
W_BAND=0.0        # R3 lever 1: winding-band consistency weight
W_SPEED=0.0       # R3 lever 3: arc-length speed-variance weight
W_WCC=0.0         # R3 lever 3-alt: CC-based per-pixel winding-label penalty
W_INTENSITY=0.0   # Path B: intensity-profile attachment weight
INTENSITY_DELTA=5.0
INTENSITY_MARGIN=0.05
W_INT_ZCOH=0.0    # Path B v2: intensity z-coherence weight
MODEL_TYPE="inr"  # "inr" (Fourier+MLP) or "perwinding" (per-winding Fourier-in-θ)
W_STRIP_Q=0.0     # Path C: strip-quality regularizer (var + HF energy)
W_DISC=0.0        # Path D: trained-discriminator weight
DISC_CKPT=""      # Path D: discriminator checkpoint path
DATA_DRIVEN_BASE=0  # Track 1: per-angle centerlines as analytical base
INIT_FROM=""        # Track 2: path to model_final.pt to warm-init residual
BSPLINE_U_PER=4     # Track 3: u-spans per winding for cubic B-spline
W_PATCH_FEAT=0.0    # Track 4: DCT patch-feature z-coh weight
W_INFONCE=0.0       # Step B: InfoNCE patch-feature contrastive weight
# MAX_SLICES is honored from the environment if set; do not redefine here.

# ── Per-variant flags ───────────────────────────────────────────────────────
case "${VARIANT}" in
  R1a) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=0 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  R1b) ATTACH=emulsion   ; SUP_STEPS=2000 ; USE_DIST=0 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  R1c) ATTACH=centerline ; SUP_STEPS=0    ; USE_DIST=0 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  R1d) ATTACH=centerline ; SUP_STEPS=2000 ; USE_DIST=0 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  R1e) ATTACH=centerline ; SUP_STEPS=2000 ; USE_DIST=0 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  R1f) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  # R1g = R1f with conformal disabled. Tests whether pure attachment on a smooth
  # distance field is enough to preserve the warm-start-learned mapping.
  R1g) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.0   ;;
  # R1h = R1f with 10× weaker conformal. Diagnoses whether collapse is caused
  # by the weight magnitude or by the structure of the loss itself.
  R1h) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.01  ;;
  # R1j = R1f with 10× lower base LR. Tests whether smoke's success was just
  # because cosine-LR put it at low LR by the time SS released. Predicts: works.
  R1j) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ;;
  # R1k = R1f with sup ratio matched to smoke (67% of total). LR at SS handoff
  # under cosine schedule is ~2.05e-4, mirroring smoke's 2.5e-4.
  R1k) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; SUP_OVERRIDE=7000 ;;
  # R1l = R1f with no warm-start. Tests whether SS losses alone, with the now-
  # corrected analytical, can find the right basin. Control for "is warm-start
  # needed at all".
  R1l) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ;;
  # R1m = R1j (lr=1e-4 winner) with the emulsion offset baked into analytical.
  # R1j had res_px=14.76 at sup end (= film→emulsion centerline distance), only
  # an "average" radial residual that fails on inner windings where dθ/du is
  # large. With the offset baked in, residual=0 is correct and INR only learns
  # tiny perturbations.
  R1m) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ;;
  # R1n = R1m but pure SS. With analytical on emulsion, attach gradient at
  # residual=0 is already minimal. Tests if warm-start is still needed once
  # the analytical itself is correct.
  R1n) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ;;
  # R1o (R2b) = R1n (pure SS, winner on clean) + learnable analytical eccentricity.
  # Adds 2 scalar params (amp, phase) that directly absorb the ecc sinusoid the
  # synthetic generator injects. Aim: close the 0.87→0.55 row_corr gap on
  # imperfect_4k without adding INR capacity.
  R1o) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; LEARN_ECC=1 ;;
  # R1p = R1m + learnable eccentricity. Warm-start gives the INR a head-start;
  # ecc params converge during SS release. Safer fallback if R1o is finicky.
  R1p) ATTACH=emulsion   ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; LEARN_ECC=1 ;;
  # R1q = R1o rerun after fixing the ecc-not-learning bug. ecc now trained by a
  # dedicated optimizer at 100× INR LR, unclipped — tests if prior R1o plateau
  # was purely an LR/clipping issue or a deeper "INR absorbs it first" issue.
  R1q) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; LEARN_ECC=1 ;;
  # R1r (R2c) = R1n + bigger INR (hidden=512, 6 layers). Tests whether the 0.55
  # ceiling on imperfect_4k is a capacity bottleneck for jitter absorption.
  R1r) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; HIDDEN_DIM=512 ; N_LAYERS_MLP=6 ;;
  # R1s (R3) = R1n + z-coherence regularizer. Synthetic geometry is z-invariant
  # by construction (jitter is computed once, applied to every slice), so the
  # learned surface should be too. Penalizing |pred_xy(u,z)-pred_xy(u,z+1)|²
  # adds the missing constraint to break the underdetermined-attachment
  # degeneracy that capped imperfect_4k at row_corr≈0.55.
  R1s) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_ZCOH=1.0 ;;
  # R1t = R1s with stronger z-coherence. Sweep alongside R1s to see if 1.0 is
  # too weak to overcome attachment noise, without rerunning a full job per weight.
  R1t) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_ZCOH=10.0 ;;
  # R1u (R3 lever 2) = R1n + per-winding (amp_k, phase_k) ecc. 56 scalar params
  # vs the global R1q's 2; targets the synthetic generator's per-winding
  # sinusoidal jitter directly. Same separate-optimizer trick as R1q.
  R1u) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; PW_ECC=1 ;;
  # R1v (R3 lever 1) = R1n + winding-band consistency. Soft-hinge penalty that
  # pred_xy radial position falls in the expected winding's [b_k, b_{k+1}] band.
  # Breaks the radial degeneracy that lets the surface hop windings.
  R1v) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_BAND=0.001 ;;
  # R1w (R3 lever 3) = R1n + arc-length speed-variance penalty. Encourages
  # uniform |∂pred_xy/∂u| so the surface advances steadily without dwelling.
  R1w) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_SPEED=1.0 ;;
  # R1x = R1v retry, w_band rescaled. R1v at 1e-3 was ~6 orders too strong
  # (band loss in px², attach in 1e-6). 1e-6 puts band same magnitude as attach.
  R1x) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_BAND=0.000001 ;;
  # R1y = R1x at 10× stronger band weight; sweep alongside R1x.
  R1y) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_BAND=0.00001 ;;
  # R1z = R1n + CC-based winding-label penalty. The label volume is built
  # from connected components on the eroded film mask, ordered by mean
  # radius, propagated to all pixels via EDT. Penalizes
  # |sampled_label − floor(u_raw)|² so the surface is pulled to the correct
  # winding rather than just to "any emulsion pixel".
  R1z) ATTACH=emulsion   ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_WCC=0.1 ;;
  # R1B1 (Path B) = R1n + intensity-profile attachment loss. δ=5px, margin=0.05.
  # Adds a true gradient inside the emulsion band: penalizes when off-surface
  # neighbors (along the local in-plane normal) are brighter than on-surface.
  # Targets the structural ceiling that mask-only attachment hits at row_corr≈0.55.
  R1B1) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INTENSITY=1.0 ;;
  # R1B2 = R1B1 with stronger intensity weight (sweep).
  R1B2) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INTENSITY=10.0 ;;
  # R1B3 = R1B1 with smaller δ (3px instead of 5px) — narrower normal probe,
  # closer to emulsion thickness so off-surface always hits film_base/air.
  R1B3) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INTENSITY=1.0 ; INTENSITY_DELTA=3.0 ;;
  # R1B4 (Path B v2) = R1n + intensity z-coherence. Penalizes
  # |I(pred(u, z_a)) - I(pred(u, z_b))|² across adjacent z. Synthetic
  # geometry is z-invariant so the SAME u across z slices should sample the
  # SAME image content. Targets the imperfect_4k angular-mismapping failure
  # mode that the radial intensity loss (R1B1-3) doesn't address.
  R1B4) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INT_ZCOH=1.0 ;;
  # R1B5 = R1B4 with stronger weight.
  R1B5) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INT_ZCOH=10.0 ;;
  # R1B6 = R1B4 + radial intensity (R1B1) combined. Both axial constraints.
  R1B6) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INT_ZCOH=1.0 ; W_INTENSITY=1.0 ;;
  # R1S = PURE SUPERVISED upper bound. SUP_STEPS = TOTAL_STEPS so the entire
  # training is MSE vs GT u-map, no SS continuation. Resolves the question
  # "can the architecture+loss family represent the answer at all?". Result
  # discriminates: ≥0.85 → SS is the only blocker; 0.6-0.8 → architecture
  # partially limits jitter expression; <0.6 → analytical+correction framing
  # is wrong, need fundamentally different model.
  R1S)  ATTACH=emulsion  ; SUP_STEPS=999999 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ;;
  # R1P_S = pure-supervised on per-winding parametric model (no shared MLP
  # weights). Tests architecture upper bound when the residual is
  # decomposed into independent per-winding Fourier-in-θ series.
  # Expected: row_corr near 1.0 since each winding's jitter has its own
  # dedicated params (no cross-winding averaging).
  R1P_S) ATTACH=emulsion ; SUP_STEPS=999999 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; MODEL_TYPE="perwinding" ;;
  # R1P_n = pure-SS on per-winding parametric model. Same SS losses as R1n
  # but new architecture. Tests whether removing shared-weight smearing
  # helps SS find the right basin (probably not — SS bottleneck is the
  # loss signal, not architecture — but cheap to verify).
  R1P_n) ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; MODEL_TYPE="perwinding" ;;
  # R1C1 (Path C) = R1n + strip-quality loss (variance + HF energy on rendered
  # strip). Differentiable proxies of the unsupervised strip_score metrics.
  # Maximize variance to prevent collapse, maximize HF energy to encourage
  # sharp frame-edge structure. Direct attack on the "wrong unwrap = featureless
  # strip" diagnosis.
  R1C1) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_STRIP_Q=1.0 ;;
  # R1C2 = R1C1 with stronger weight (sweep).
  R1C2) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_STRIP_Q=10.0 ;;
  # R1C3 = R1C1 with even stronger weight, see if signal can dominate.
  R1C3) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_STRIP_Q=100.0 ;;
  # R1D1 (Path D) = R1n + trained CNN discriminator. Discriminator was
  # trained on synthetic clean GT strips vs corruptions (shuffles, blurs,
  # jitter). Provides high-dimensional learned feature signal that aggregate
  # losses cannot. Weight 1.0 starting point.
  R1D1) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=1.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D2 = R1D1 with stronger weight (sweep).
  R1D2) ATTACH=emulsion  ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=10.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D3 = R1D1 + warm-start (3000 sup steps then SS+disc). Start near GT
  # so discriminator gradient has clearer signal, not orthogonal noise.
  R1D3) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=1.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D4 = R1D3 with disc weight=2 (slight increase, find optimal balance).
  R1D4) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=2.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D5 = R1D3 with disc weight=5.
  R1D5) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=5.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D6 = R1D3 + stronger conformal (rebalance after disc loss showed
  # tendency to break conformal in R1D3 — E vs G diverged).
  R1D6) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.5   ; LR="1e-4" ; W_DISC=1.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_clean_4k/strip_discriminator.pt" ;;
  # R1D7 = R1D3 with discriminator trained on rendered-at-GT patches.
  # Fixes the distribution-shift bug that made R1D3/D5 full collapse:
  # the original CNN was trained on source [0,1] strip; this one trains on
  # CT-intensity-space strips matching what the unwrapping actually produces.
  R1D7) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=1.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_imperfect_4k_rendered_at_gt/strip_discriminator.pt" ;;
  # R1D8 = R1D7 with disc weight=5 (smoke-winner R1D5 had this).
  R1D8) ATTACH=emulsion  ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_DISC=5.0 ; DISC_CKPT="/data/user/li_k1/M_thesis/unwrapping/inr/results/discriminator_imperfect_4k_rendered_at_gt/strip_discriminator.pt" ;;
  # ── Round 9 / Track 1 ────────────────────────────────────────────────────
  # R1T1 = R1n (pure SS winner) + data-driven analytical base. Per-angle
  # centerlines from raycast replace concentric circles. Each winding becomes
  # its own non-circular curve, absorbing eccentricity and a fraction of
  # jitter into the base so the residual carries less. Expected: SS
  # imperfect_4k 0.55 → 0.7+, video_4k_imperfect 0.64 → 0.7+.
  R1T1)   ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; DATA_DRIVEN_BASE=1 ;;
  # R1T1_S = R1S (pure supervised) + data-driven base. Tests whether the
  # architecture cap (0.91 on video imperfect) lifts when the base absorbs
  # the eccentricity term the residual previously had to learn.
  R1T1_S) ATTACH=emulsion ; SUP_STEPS=999999 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; DATA_DRIVEN_BASE=1 ;;
  # ── Round 9 / Track 2 ────────────────────────────────────────────────────
  # R1T2 = R1n on imperfect, but residual weights warm-initialized from the
  # corresponding clean-preset R1n checkpoint. INIT_FROM env var must be set
  # by the caller (or paired-job submitter): e.g. clean → imperfect chain on
  # `video_4k` → `video_4k_imperfect`. Expected: +0.03 to +0.08 over cold-SS.
  R1T2)   ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ;;
  # R1T12 = Track 1 + Track 2 stacked: data-driven base AND warm-init.
  R1T12)  ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; DATA_DRIVEN_BASE=1 ;;
  # ── Round 9 / Track 3 ────────────────────────────────────────────────────
  # R1B_S = B-spline tensor-product residual, pure supervised. Tests whether
  # replacing the INR's shared-MLP smoothness prior with cubic-B-spline
  # control points lifts the architecture cap above 0.91 on video imperfect.
  # Default geometry: 4 u-spans/winding (=112 u-spans on 28 windings) × 3
  # z-spans, ~720 control points × 2-d ≈ 1440 params (vs INR's 460k).
  R1B_S)  ATTACH=emulsion ; SUP_STEPS=999999 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; MODEL_TYPE="bspline" ;;
  # R1B_n = B-spline + pure SS. Same SS losses as R1n. Lower-priority test —
  # SS bottleneck is loss signal, not architecture, but cheap to verify the
  # B-spline doesn't make SS worse than the INR's 0.64.
  R1B_n)  ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; MODEL_TYPE="bspline" ;;
  # R1B_S12 = R1B_S with 12 u-spans/winding (vs default 4). Tests whether the
  # 0.62 ceiling on smoke is B-spline-capacity-limited (high-freq jitter
  # needs finer knots) or a deeper generalization issue.
  R1B_S12) ATTACH=emulsion ; SUP_STEPS=999999 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; MODEL_TYPE="bspline" ; BSPLINE_U_PER=12 ;;
  # ── Track 4 — DCT patch-feature z-coherence ──────────────────────────────
  # R1PF1 = R1n cold-SS + patch-feature loss w=1. Risk of Goodhart-collapse
  # from random init, but cheap to verify.
  R1PF1)  ATTACH=emulsion ; SUP_STEPS=0      ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_PATCH_FEAT=1.0 ;;
  # R1PF1ws = warm-start (3000 sup) + patch-feature w=1. Mirrors the R1D3
  # recipe that worked for the discriminator at smoke length. Prior places
  # surface in non-degenerate region first, then patch-feat refines.
  R1PF1ws) ATTACH=emulsion ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_PATCH_FEAT=1.0 ;;
  # R1PF10ws = R1PF1ws with stronger weight.
  R1PF10ws) ATTACH=emulsion ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_PATCH_FEAT=10.0 ;;
  # Step B — InfoNCE contrastive patch-feature on synthetic.
  # R1NCE1ws: warm-start (3000 sup) + InfoNCE w=1. Mirrors R1PF1ws.
  R1NCE1ws) ATTACH=emulsion ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INFONCE=1.0 ;;
  # R1NCE1n: pure SS + InfoNCE (no warm-start) — tests whether InfoNCE's
  # collapse-immunity allows it to work without an anchor (unlike R1PF).
  R1NCE1n)  ATTACH=emulsion ; SUP_STEPS=0    ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INFONCE=1.0 ;;
  # R1NCE5ws: stronger InfoNCE weight.
  R1NCE5ws) ATTACH=emulsion ; SUP_STEPS=3000 ; USE_DIST=1 ; W_RES_REG=0.0  ; W_CONF=0.1   ; LR="1e-4" ; W_INFONCE=5.0 ;;
  *)   echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

# ── Per-mode budgets ────────────────────────────────────────────────────────
# Partition & wall-time are set via the `submit_R1.sh` launcher's sbatch flags.
case "${MODE}" in
  smoke)
    TOTAL_STEPS=1500
    RAMP=500
    EVAL_REAL=0
    CKPT_EVERY=0
    ;;
  full)
    TOTAL_STEPS=10000
    RAMP=2000
    EVAL_REAL=1
    CKPT_EVERY=500   # save 20 intermediate ckpts for sweep eval
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

# Per-variant SUP override (e.g. R1k tests smoke ratio at full budget).
if [ -n "${SUP_OVERRIDE}" ]; then
  SUP_STEPS="${SUP_OVERRIDE}"
fi

# In warm-start variants, SUP_STEPS is counted against TOTAL_STEPS.
# Ensure at least 500 SS steps remain for release (smoke budget is 1500).
# R1S intentionally runs full supervision — cap SUP_STEPS at TOTAL_STEPS instead
# of leaving SS room.
if [ "${VARIANT}" = "R1S" ] || [ "${VARIANT}" = "R1P_S" ]; then
  SUP_STEPS="${TOTAL_STEPS}"
elif [ "${SUP_STEPS}" -gt 0 ] && [ "${MODE}" = "smoke" ]; then
  MAX_SUP=$((TOTAL_STEPS - 500))
  if [ "${SUP_STEPS}" -gt "${MAX_SUP}" ]; then
    SUP_STEPS="${MAX_SUP}"
  fi
fi

if [ "${PRESET}" = "clean_4k" ]; then
  OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1_${MODE}/${VARIANT}"
else
  OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1_${MODE}_${PRESET}/${VARIANT}"
fi
TRAIN_DIR="${OUT_BASE}/train"
EVAL_SYN_DIR="${OUT_BASE}/eval_synthetic"
EVAL_REAL_DIR="${OUT_BASE}/eval_real"

# ── Environment ─────────────────────────────────────────────────────────────
mkdir -p logs "${TRAIN_DIR}" "${EVAL_SYN_DIR}"
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  Round 1 — variant ${VARIANT} / mode ${MODE}"
echo "============================================================"
echo "  attachment   : ${ATTACH}"
echo "  use_distance : ${USE_DIST}"
echo "  w_res_reg    : ${W_RES_REG}"
echo "  sup_steps    : ${SUP_STEPS}"
echo "  total_steps  : ${TOTAL_STEPS}"
echo "  ramp_steps   : ${RAMP}"
echo "  base lr      : ${LR}"
echo "  w_z_coherence: ${W_ZCOH}"
echo "  pw_ecc       : ${PW_ECC}"
echo "  w_band       : ${W_BAND}"
echo "  w_arc_speed  : ${W_SPEED}"
echo "  w_winding_cc : ${W_WCC}"
echo "  preset       : ${PRESET}"
echo "  synthetic    : ${SYN_DIR}"
echo "  real         : ${REAL_DIR} (eval=${EVAL_REAL})"
echo "  out          : ${OUT_BASE}"
echo "  job id       : ${SLURM_JOB_ID}"
echo "  partition    : ${SLURM_JOB_PARTITION:-unset}"
echo "  started      : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

# ── Train on synthetic clean_4k ─────────────────────────────────────────────
echo ""
echo "--- [1/3] Training on clean_4k ---"
EXTRA_FLAGS=""
if [ "${USE_DIST}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --use-distance-attach"
fi
if [ "${LEARN_ECC}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --learn-eccentricity"
fi
# Use awk for fractional comparison (bash [ ] only handles ints).
if awk "BEGIN{exit !(${W_ZCOH} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-z-coherence ${W_ZCOH}"
fi
if [ "${PW_ECC}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --per-winding-eccentricity"
fi
if awk "BEGIN{exit !(${W_BAND} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-winding-band ${W_BAND}"
fi
if awk "BEGIN{exit !(${W_SPEED} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-arc-speed ${W_SPEED}"
fi
if awk "BEGIN{exit !(${W_WCC} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-winding-cc ${W_WCC}"
fi
if awk "BEGIN{exit !(${W_INTENSITY} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-intensity ${W_INTENSITY} --intensity-delta-px ${INTENSITY_DELTA} --intensity-margin ${INTENSITY_MARGIN}"
fi
if awk "BEGIN{exit !(${W_INT_ZCOH} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-intensity-zcoh ${W_INT_ZCOH}"
fi
if awk "BEGIN{exit !(${W_STRIP_Q} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-strip-quality ${W_STRIP_Q}"
fi
if awk "BEGIN{exit !(${W_DISC} > 0)}" && [ -n "${DISC_CKPT}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --discriminator-ckpt ${DISC_CKPT} --w-discriminator ${W_DISC}"
fi
if [ "${DATA_DRIVEN_BASE}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --data-driven-base"
fi
if awk "BEGIN{exit !(${W_PATCH_FEAT} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-patch-feature-zcoh ${W_PATCH_FEAT}"
fi
if awk "BEGIN{exit !(${W_INFONCE} > 0)}"; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --w-patch-feature-infonce ${W_INFONCE}"
fi
# Honor MAX_SLICES from env (may also be set per variant above).
if [ -n "${MAX_SLICES:-}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --max-slices ${MAX_SLICES}"
  echo "  max_slices  : ${MAX_SLICES}"
fi
# INIT_FROM may also be passed via env (caller submits R1T2 with INIT_FROM=/path).
if [ -n "${INIT_FROM:-}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --init-from ${INIT_FROM}"
  echo "  init_from   : ${INIT_FROM}"
fi

srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${TRAIN_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --supervised-steps "${SUP_STEPS}" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    --steps "${TOTAL_STEPS}" \
    --ramp-steps "${RAMP}" \
    --batch-size 131072 \
    --lr "${LR}" \
    --w-attach 1.0 \
    --w-conformal "${W_CONF}" \
    --w-res-reg "${W_RES_REG}" \
    --sigma 20.0 \
    --hidden-dim "${HIDDEN_DIM}" \
    --n-layers-mlp "${N_LAYERS_MLP}" \
    --model-type "${MODEL_TYPE}" \
    --bspline-u-per-winding "${BSPLINE_U_PER}" \
    --log-every 50 \
    --ckpt-every "${CKPT_EVERY}" \
    ${EXTRA_FLAGS}

# ── Eval on synthetic (with GT metrics) ─────────────────────────────────────
echo ""
echo "--- [2/3] Eval on synthetic (with GT metrics) ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${SYN_DIR}" \
    --out-dir  "${EVAL_SYN_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --ckpt   "${TRAIN_DIR}/model_final.pt" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    $( [ "${DATA_DRIVEN_BASE}" = "1" ] && echo "--data-driven-base" ) \
    $( [ -n "${MAX_SLICES:-}" ] && echo "--max-slices ${MAX_SLICES}" )

# ── Qualitative transfer check on real Mickey ──────────────────────────────
# The INR was trained with synthetic geometry; running it directly on real
# data produces a strip only if we re-detect geometry from real. We re-init
# the dataset from real, and load the weights — the INR learns f(u,z)→(x,y),
# which depends on the *detected* boundaries (analytical base). So results
# reflect how well the INR generalises to a different, re-detected spiral.
if [ "${EVAL_REAL}" = "1" ]; then
  mkdir -p "${EVAL_REAL_DIR}"
  echo ""
  echo "--- [3/3] Qualitative eval on real Mickey ---"
  # --max-slices 20 caps GPU mem: real Mickey has ~500 slices @ 3063×3062
  # → 18GB per volume × 3 (mask, image, dist) = 56GB on GPU, OOMs the 80GB A100
  # when combined with model + autograd state. 20 slices is enough for the
  # qualitative transfer check.
  srun python -u -m unwrapping.inr.surface_eval \
      --data-dir "${REAL_DIR}" \
      --out-dir  "${EVAL_REAL_DIR}" \
      --max-slices 20 \
      --attachment "${ATTACH}" \
      --centerline-erode 1 \
      --ckpt "${TRAIN_DIR}/model_final.pt" \
      $( [ "${DATA_DRIVEN_BASE}" = "1" ] && echo "--data-driven-base" )
fi

echo ""
echo "=== R1 ${VARIANT}/${MODE} complete at $(date) ==="
