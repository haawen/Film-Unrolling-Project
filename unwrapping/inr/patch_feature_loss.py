"""
Patch-feature self-supervised loss (Track 4).

Renders small strip patches from the model's current `pred_xy`, computes a
fixed (non-trainable) feature representation per patch, and penalizes the
feature-vector distance between patches sampled at the SAME u from
ADJACENT z-slices. The synthetic geometry is z-invariant by construction;
real Mickey's geometry varies slowly with z. In both cases, the same u
should sample the same physical film point across z.

Why a fixed feature extractor (not a trainable discriminator):
- Round 7's CNN discriminator approach worked at smoke length but always
  collapsed at full length via Goodhart's law: any trained CNN reaches 100%
  accuracy on its training set quickly and becomes gameable by adversarial
  surfaces.
- A non-trainable transform (DCT, Gabor) has no parameters to game. Two
  patches with the same physical content produce similar features
  deterministically; uniform-content patches produce near-zero feature
  vectors that have UNDEFINED cosine direction — we use cosine distance
  with epsilon-clamped norms so the trivial uniform-collapse mode does not
  give a low loss.

The loss has TWO terms (sum, equal-weighted by default):
1. cosine-distance term:  1 - cos_sim(feat_a, feat_b)   — alignment.
2. magnitude floor:       max(0, target_norm - ‖feat‖)² — penalizes
   collapse to uniform patches (which give near-zero feature vectors).
   Without this, the model would minimize the cosine term by drifting to
   regions with no content (R1B4-style collapse).

Usage from surface_train.py:
    from unwrapping.inr.patch_feature_loss import compute_patch_feature_zcoh

    if args.w_patch_feature_zcoh > 0:
        l_patch = compute_patch_feature_zcoh(
            dataset, model, args.disc_n_patches,
            patch_w=args.patch_feature_w,
            n_dct_keep=args.patch_feature_n_dct,
            target_norm=args.patch_feature_target_norm,
        )
        loss = loss + args.w_patch_feature_zcoh * l_patch
"""

import math

import torch
import torch.nn.functional as F


def _dct_matrix(n, device, dtype=torch.float32):
    """DCT-II matrix (orthonormal): produces feature[k] = sum_n x[n]·cos(...).

    Returns a (n, n) matrix `M` such that `M @ x` gives the DCT of `x` (1D).
    Cached on the module-level via lru_cache wouldn't help (device varies);
    callers should cache on the dataset object if needed.
    """
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # (1, n)
    n_idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n, 1)
    M = torch.cos(math.pi * (2 * n_idx + 1) * k / (2 * n))
    M[0] *= 1.0 / math.sqrt(n)
    M[1:] *= math.sqrt(2.0 / n)
    return M.T  # (n, n) so that out = M @ x (rows = freqs, cols = samples)


def compute_patch_feature_infonce(
    dataset, model, n_anchors=64, patch_w=64, n_dct_keep=16,
    temperature=0.1, eps=1e-6,
):
    """Step B — InfoNCE contrastive patch-feature loss.

    Forbids the trivial-collapse mode that beat the pairwise version (Step A
    failure): if pred_xy(u, z) becomes constant, all batch patches converge
    to the same feature vector, the similarity matrix becomes uniform, and
    InfoNCE explodes to log(n_anchors). Penalizes collapse by construction.

    Mechanism (SimCLR-style):
      - Sample n_anchors random (u_start, z_a) pairs, z_b = z_a + 1.
      - For each anchor: render patch_a from pred_xy(u, z_a) and patch_b
        from pred_xy(u, z_b), DCT each row, drop DC, mean-pool across z to
        produce a single feature vector per anchor per side.
      - Build (n_anchors, n_anchors) similarity matrix `S[i, j] =
        cos_sim(feat_a[i], feat_b[j])`.
      - Diagonal = positives (same u, adjacent z); off-diagonal = negatives
        (different u within batch).
      - Cross-entropy with diagonal target: positives must be more similar
        than negatives.

    Args:
        dataset: SurfaceDataset.
        model: residual model.
        n_anchors: batch size. Larger = more negatives = harder task =
            stronger signal. 64 is a good default; 128+ may help.
        patch_w: u-width of each patch in strip pixels.
        n_dct_keep: number of mid-frequency DCT coefficients kept (after DC).
        temperature: softmax sharpness. Lower = harder contrast.

    Returns:
        scalar loss tensor.
    """
    if dataset.Z < 2:
        return torch.zeros((), device=dataset.device)

    device = dataset.device
    n_layers = dataset.n_layers
    Z = dataset.Z

    # Sample anchor positions.
    px_per_winding = max(1.0, dataset.total_arc / max(1, n_layers))
    u_extent = patch_w / px_per_winding
    u_extent = min(u_extent, n_layers - 1e-3)

    u_starts = torch.rand(n_anchors, device=device) * (n_layers - u_extent)
    u_offsets = torch.linspace(0.0, u_extent, patch_w, device=device)
    u_grid = u_starts.unsqueeze(1) + u_offsets.unsqueeze(0)   # (n_anchors, patch_w)
    u_flat = u_grid.reshape(-1)                                # (n_anchors * patch_w,)

    z_a = torch.randint(0, Z - 1, (n_anchors,), device=device)
    z_b = z_a + 1
    z_a_flat = z_a.unsqueeze(1).expand(-1, patch_w).reshape(-1)
    z_b_flat = z_b.unsqueeze(1).expand(-1, patch_w).reshape(-1)

    # Render via model + sample_image.
    def _render(u_flat_, z_idx_flat_):
        u_norm = u_flat_ / n_layers * 2.0 - 1.0
        z_norm = z_idx_flat_.float() / max(1, Z - 1) * 2.0 - 1.0
        uv = torch.stack([u_norm, z_norm], dim=1)
        base = dataset.analytical_xy_norm(u_flat_)
        res = model(uv)
        pred_xy = base + res
        I = dataset.sample_image(pred_xy, z_idx_flat_)
        return I.view(n_anchors, patch_w)

    patches_a = _render(u_flat, z_a_flat)   # (n_anchors, patch_w)
    patches_b = _render(u_flat, z_b_flat)

    # 1D DCT, drop DC, keep mid-freq.
    dct = _dct_matrix(patch_w, device=device)               # (patch_w, patch_w)
    feats_a = patches_a @ dct.T                              # (n_anchors, patch_w)
    feats_b = patches_b @ dct.T
    feats_a = feats_a[:, 1:1 + n_dct_keep]
    feats_b = feats_b[:, 1:1 + n_dct_keep]

    # L2-normalize for cosine similarity.
    feats_a = F.normalize(feats_a, p=2, dim=1, eps=eps)
    feats_b = F.normalize(feats_b, p=2, dim=1, eps=eps)

    # Similarity matrix (n_anchors, n_anchors).  S[i, j] = <a_i, b_j>.
    sim = feats_a @ feats_b.T / temperature                  # (n_anchors, n_anchors)

    # InfoNCE: each row's positive is the diagonal entry, negatives are the
    # rest of the row. Cross-entropy with target indices = arange(n).
    target = torch.arange(n_anchors, device=device)
    loss_ab = F.cross_entropy(sim, target)
    # Symmetric (b → a as well) for stability.
    loss_ba = F.cross_entropy(sim.T, target)
    return 0.5 * (loss_ab + loss_ba)


def compute_patch_feature_zcoh(
    dataset, model, n_patches, patch_w=64, n_dct_keep=16,
    target_norm=0.05, eps=1e-6,
):
    """SS loss: feature-space z-coherence between adjacent-z patches.

    Args:
        dataset: SurfaceDataset (provides analytical_xy_norm, sample_image,
            n_layers, total_arc, Z, device).
        model: residual deformation model.
        n_patches: number of (u_start, z_a) pairs to sample.
        patch_w: patch width in u-pixels (strip pixels per patch).
        n_dct_keep: number of mid-frequency DCT coefficients to keep
            (excluding DC). Default 16 is a reasonable structural-content
            proxy at patch_w=64.
        target_norm: floor on the mean L2 norm of feature vectors (in
            normalized intensity units). Patches with weaker content than
            this are penalized as "near-collapse". 0.05 ≈ patches with
            visible frame structure.
        eps: numerical safety for L2 normalization.

    Returns:
        scalar loss tensor.
    """
    if dataset.Z < 2:
        return torch.zeros((), device=dataset.device)

    device = dataset.device
    n_layers = dataset.n_layers
    Z = dataset.Z

    # --- Sample (u_start, z_a) pairs -------------------------------------
    # u_extent per patch: assume strip ~ total_arc px, n_layers windings →
    # px-per-winding ≈ total_arc / n_layers. patch_w in strip-pixels covers
    # patch_w / px_per_winding windings.
    px_per_winding = max(1.0, dataset.total_arc / max(1, n_layers))
    u_extent = patch_w / px_per_winding
    u_extent = min(u_extent, n_layers - 1e-3)

    u_starts = torch.rand(n_patches, device=device) * (n_layers - u_extent)
    u_offsets = torch.linspace(0.0, u_extent, patch_w, device=device)
    # (n_patches, patch_w)
    u_grid = u_starts.unsqueeze(1) + u_offsets.unsqueeze(0)
    u_flat = u_grid.reshape(-1)  # (n_patches * patch_w,)

    z_a = torch.randint(0, Z - 1, (n_patches,), device=device)
    z_b = z_a + 1
    # broadcast over patch_w → (n_patches * patch_w,)
    z_a_flat = z_a.unsqueeze(1).expand(-1, patch_w).reshape(-1)
    z_b_flat = z_b.unsqueeze(1).expand(-1, patch_w).reshape(-1)

    # --- Render patches via model + sample_image -------------------------
    def _render(u_flat_, z_idx_flat_):
        u_norm = u_flat_ / n_layers * 2.0 - 1.0
        z_norm = z_idx_flat_.float() / max(1, Z - 1) * 2.0 - 1.0
        uv = torch.stack([u_norm, z_norm], dim=1)
        base = dataset.analytical_xy_norm(u_flat_)
        res = model(uv)
        pred_xy = base + res
        I = dataset.sample_image(pred_xy, z_idx_flat_)
        return I.view(n_patches, patch_w)

    patches_a = _render(u_flat, z_a_flat)  # (n_patches, patch_w)
    patches_b = _render(u_flat, z_b_flat)

    # --- DCT features (drop DC, keep n_dct_keep coefficients) -----------
    dct = _dct_matrix(patch_w, device=device)  # (patch_w, patch_w)
    feats_a = patches_a @ dct.T  # (n_patches, patch_w)
    feats_b = patches_b @ dct.T
    feats_a = feats_a[:, 1:1 + n_dct_keep]  # drop DC, keep next K
    feats_b = feats_b[:, 1:1 + n_dct_keep]

    # --- Cosine distance + magnitude floor ------------------------------
    norm_a = feats_a.norm(dim=1)
    norm_b = feats_b.norm(dim=1)
    feats_a_unit = feats_a / (norm_a.unsqueeze(1) + eps)
    feats_b_unit = feats_b / (norm_b.unsqueeze(1) + eps)
    cos_sim = (feats_a_unit * feats_b_unit).sum(dim=1)
    l_align = (1.0 - cos_sim).mean()

    mean_norm = 0.5 * (norm_a + norm_b)
    l_floor = (F.relu(target_norm - mean_norm) ** 2).mean()

    return l_align + l_floor


def compute_frame_periodicity_loss(
    dataset, model, n_patches, patch_w=64, n_dct_keep=16,
    frame_period_u=0.0665, target_norm=0.05, eps=1e-6,
):
    """A1 — frame-period periodicity on the implied strip.

    Hypothesis: the unrolled strip is approximately periodic at frame_width
    in u (every frame_period_u winding-fractions of arc-length). The implied
    strip I(u, z) = CT[ f_θ(u, z), z ] should therefore satisfy
    I(u, z) ≈ I(u + frame_period_u, z) in feature space.

    Loss shape identical to `compute_patch_feature_zcoh` (DCT mid-freq cosine
    distance + magnitude floor) but with the pair offset along u instead of
    along z:
        anchor:    patch at (u, z)
        positive:  patch at (u + frame_period_u, z)

    Why this is not the closed DCT z-coh family:
      - The pairing rule is a KNOWN PHYSICAL PERIOD (frame pitch), not a
        learned or data-driven adjacency. The drift modes that beat
        z-coh-on-real (move pred_xy to flat content, collapse to constant)
        either violate the magnitude floor (they don't) or have to do so
        while *also* matching the period exactly across hundreds of frames.
      - Cosine sim + magnitude floor are kept (collapse-immunity argument
        unchanged from zcoh case).

    Failure modes to watch for:
      - Wrong period: if frame_period_u disagrees with the data, the loss
        either is unsatisfiable (no improvement) or worse, drives pred_xy
        toward periodic-at-wrong-period — which is a structural failure
        and will look like row_corr ≤ baseline. Sanity: compute the period
        from generator params; don't fit it.
      - Scene cuts / bright-frame gradient: hard transitions in scene
        content break exact periodicity. Mitigation: many random samples
        average out; we don't require exact match, only cos sim > 0.

    Args:
        n_patches: number of (u, z) anchors per step.
        patch_w: strip-pixel width of each patch.
        n_dct_keep: number of mid-freq DCT coefficients (excluding DC).
        frame_period_u: u-distance between adjacent frames. Compute as
            frame_width_px / total_arc_px * n_layers. For HQ video presets
            (n_z=256, frame_w=341 px, total_arc≈143558, n_layers=28),
            frame_period_u ≈ 0.0665.
        target_norm: floor on feature L2 norm (anti-collapse).
    """
    device = dataset.device
    n_layers = dataset.n_layers
    Z = dataset.Z

    px_per_winding = max(1.0, dataset.total_arc / max(1, n_layers))
    u_extent = patch_w / px_per_winding
    u_extent = min(u_extent, n_layers - frame_period_u - 1e-3)
    if u_extent <= 0:
        return torch.zeros((), device=device)

    # Sample anchor (u_start, z) — anchor patch spans [u_start, u_start + u_extent].
    # Positive patch is shifted by frame_period_u in u, same z.
    # Restrict u_start so the positive patch stays in-domain.
    max_u_start = n_layers - u_extent - frame_period_u - 1e-3
    if max_u_start <= 0:
        return torch.zeros((), device=device)
    u_starts = torch.rand(n_patches, device=device) * max_u_start
    u_offsets = torch.linspace(0.0, u_extent, patch_w, device=device)
    u_grid_a = u_starts.unsqueeze(1) + u_offsets.unsqueeze(0)
    u_grid_b = u_grid_a + frame_period_u
    u_flat_a = u_grid_a.reshape(-1)
    u_flat_b = u_grid_b.reshape(-1)

    z_idx = torch.randint(0, Z, (n_patches,), device=device)
    z_flat = z_idx.unsqueeze(1).expand(-1, patch_w).reshape(-1)

    def _render(u_flat_, z_idx_flat_):
        u_norm = u_flat_ / n_layers * 2.0 - 1.0
        z_norm = z_idx_flat_.float() / max(1, Z - 1) * 2.0 - 1.0
        uv = torch.stack([u_norm, z_norm], dim=1)
        base = dataset.analytical_xy_norm(u_flat_)
        res = model(uv)
        pred_xy = base + res
        I = dataset.sample_image(pred_xy, z_idx_flat_)
        return I.view(n_patches, patch_w)

    # Render BOTH intensity (for DCT features) and mask (for on-emulsion
    # weighting). Without mask-weighting the model trivially satisfies
    # periodicity by drifting pred_xy off the emulsion onto the uniform
    # film-base bands (which repeat radially) — same failure mode as the
    # closed DCT z-coh family. With mask-weighting, off-emulsion patches
    # contribute 0 to A1, so attachment+conformal continue to dominate.
    def _render_with_mask(u_flat_, z_idx_flat_):
        u_norm = u_flat_ / n_layers * 2.0 - 1.0
        z_norm = z_idx_flat_.float() / max(1, Z - 1) * 2.0 - 1.0
        uv = torch.stack([u_norm, z_norm], dim=1)
        base = dataset.analytical_xy_norm(u_flat_)
        res = model(uv)
        pred_xy = base + res
        I = dataset.sample_image(pred_xy, z_idx_flat_)
        M = dataset.sample_mask(pred_xy, z_idx_flat_)
        return I.view(n_patches, patch_w), M.view(n_patches, patch_w)

    patches_a, mask_a = _render_with_mask(u_flat_a, z_flat)
    patches_b, mask_b = _render_with_mask(u_flat_b, z_flat)

    # Per-patch on-emulsion weight: mean mask over the patch, clipped to [0,1].
    # Detached so the mask-weight doesn't pull pred_xy via the attachment
    # surface (that's the job of the attachment loss; here we just gate).
    w_a = mask_a.mean(dim=1).clamp(0.0, 1.0).detach()
    w_b = mask_b.mean(dim=1).clamp(0.0, 1.0).detach()
    w_pair = w_a * w_b                                       # (n_patches,)
    # Normalize so the loss magnitude is comparable across batches regardless
    # of how many patches happened to land on-emulsion.
    w_sum = w_pair.sum().clamp(min=1.0)

    dct = _dct_matrix(patch_w, device=device)
    feats_a = patches_a @ dct.T
    feats_b = patches_b @ dct.T
    feats_a = feats_a[:, 1:1 + n_dct_keep]
    feats_b = feats_b[:, 1:1 + n_dct_keep]

    norm_a = feats_a.norm(dim=1)
    norm_b = feats_b.norm(dim=1)
    feats_a_unit = feats_a / (norm_a.unsqueeze(1) + eps)
    feats_b_unit = feats_b / (norm_b.unsqueeze(1) + eps)
    cos_sim = (feats_a_unit * feats_b_unit).sum(dim=1)
    l_align = ((1.0 - cos_sim) * w_pair).sum() / w_sum

    mean_norm = 0.5 * (norm_a + norm_b)
    l_floor = ((F.relu(target_norm - mean_norm) ** 2) * w_pair).sum() / w_sum

    return l_align + l_floor
