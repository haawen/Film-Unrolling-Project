"""
StripDiscriminator: small 2D CNN trained to classify "real movie strip patches"
vs corrupted ones. Used as a learned similarity loss on the predicted strip
during unwrapping training (Vesuvius-style ink-detector approach).

Why: every aggregate-metric SS loss we tried (variance, HF energy, intensity
z-coherence, etc.) was satisfied trivially by the unwrapping model in ways
that didn't correspond to correct unwrapping. A CNN trained on examples
learns high-dimensional features — frame-edge co-occurrence patterns,
perforation periodicity, content texture — that can't be satisfied by
trivial pred_xy drift.

Pipeline:
    1. Load synthetic clean_4k GT strip (provides "real" examples)
    2. Extract random (n_z, patch_w) patches as POSITIVES
    3. Apply structural corruptions (per-row shuffles, blurs, sub-frame
       offsets, intensity noise) for NEGATIVES
    4. Train BCE classifier: real (1.0) vs corrupted (0.0)
    5. Save weights → consumed by surface_train.py via --discriminator-ckpt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Architecture ──────────────────────────────────────────────────────────


class StripDiscriminator(nn.Module):
    """2D CNN classifier for strip patches.

    Input: (B, 1, n_z, patch_w), values in roughly [0, 1].
    Output: (B,) sigmoid probability that the patch is "real movie content".
    """

    def __init__(self, n_z=20, patch_w=64, base_channels=16):
        super().__init__()
        c = base_channels
        # Use stride-1 convs in z (only 20 rows) and stride-2 pools in u.
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),                # u: 64→32
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),                # z: 20→10, u: 32→16
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),                # z: 10→5, u: 16→8
            nn.Conv2d(c * 4, c * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Linear(c * 4, 1),
        )

    def forward(self, x):
        feat = self.features(x).view(x.size(0), -1)
        logits = self.classifier(feat).squeeze(-1)
        return torch.sigmoid(logits), logits

    def score(self, x):
        """Return only the sigmoid probability (for use as a loss)."""
        return self.forward(x)[0]


# ─── Patch extraction & corruption ─────────────────────────────────────────


def extract_random_patches(strip, batch_size, patch_w, rng=None):
    """Extract (batch_size, n_z, patch_w) random patches from `strip`.

    strip: (n_z, arc_length) numpy array, values in [0, 1].
    """
    if rng is None:
        rng = np.random
    n_z, L = strip.shape
    starts = rng.randint(0, L - patch_w, size=batch_size)
    out = np.empty((batch_size, n_z, patch_w), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = strip[:, s:s + patch_w]
    return out


def corrupt_patches(patches, mode, rng=None):
    """Apply structural corruption to a batch of patches.

    Modes (each mimics a way the unwrapping could be wrong):
        'shuffle_u'   : random per-row reorder of u columns (kills coherence)
        'shuffle_z'   : permute z-rows (z-coherence destroyed)
        'blur'        : strong horizontal Gaussian blur (kills frame edges)
        'jitter'      : per-row random small u-offset (mimics jitter-induced
                         angular drift between rows; this is the *target*
                         failure mode we want CNN to detect)
        'noise'       : additive gaussian noise (lower than typical signal)
        'mix'         : random mix of the above per sample
    """
    if rng is None:
        rng = np.random
    p = patches.copy()
    B, Z, W = p.shape

    if mode == 'shuffle_u':
        for i in range(B):
            for z in range(Z):
                idx = rng.permutation(W)
                p[i, z] = p[i, z, idx]
    elif mode == 'shuffle_z':
        for i in range(B):
            idx = rng.permutation(Z)
            p[i] = p[i, idx]
    elif mode == 'blur':
        # Heavy horizontal Gaussian blur
        kernel_w = 11
        sigma = 4.0
        x = np.arange(kernel_w) - kernel_w // 2
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        for i in range(B):
            for z in range(Z):
                p[i, z] = np.convolve(p[i, z], kernel, mode='same')
    elif mode == 'jitter':
        # Per-z random small offset of the u-pattern (mimics imperfect
        # angular alignment that breaks z-coherence).
        # Roll shifts each row independently by ±max_shift pixels.
        max_shift = max(2, W // 16)
        for i in range(B):
            base = p[i, Z // 2].copy()
            for z in range(Z):
                shift = rng.randint(-max_shift, max_shift + 1)
                # Use the central row's content shifted, so neighboring rows
                # disagree at the same physical u — the actual failure pattern.
                p[i, z] = np.roll(base, shift)
    elif mode == 'noise':
        std = 0.15
        p = np.clip(p + rng.normal(0, std, p.shape).astype(np.float32), 0, 1)
    elif mode == 'mix':
        out = np.empty_like(p)
        for i in range(B):
            m = rng.choice(
                ['shuffle_u', 'shuffle_z', 'blur', 'jitter', 'noise']
            )
            out[i] = corrupt_patches(p[i:i + 1], m, rng=rng)[0]
        return out
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return p


_CORRUPTION_MODES = ['shuffle_u', 'shuffle_z', 'blur', 'jitter', 'noise']


def build_batch(strip, batch_size, patch_w, rng=None, batch_idx=0):
    """Return (x, y): half real (label=1), half corrupted (label=0).

    Corruption mode cycles deterministically across batches for fast training
    (avoids per-sample recursion in 'mix' mode that was triggering hangs).
    """
    if rng is None:
        rng = np.random
    half = batch_size // 2
    pos = extract_random_patches(strip, half, patch_w, rng=rng)
    pos_pre = extract_random_patches(strip, half, patch_w, rng=rng)
    mode = _CORRUPTION_MODES[batch_idx % len(_CORRUPTION_MODES)]
    neg = corrupt_patches(pos_pre, mode, rng=rng)

    x = np.concatenate([pos, neg], axis=0)[:, None, :, :]  # (B, 1, n_z, patch_w)
    y = np.concatenate([
        np.ones(half, dtype=np.float32),
        np.zeros(half, dtype=np.float32),
    ])
    perm = rng.permutation(batch_size)
    return x[perm], y[perm]


# ─── Training ──────────────────────────────────────────────────────────────


def render_strip_from_gt(image_volume, u_map, seg_2d, n_u_bins):
    """Render the strip in CT-intensity space by sampling the volume at GT
    u-map positions. Each strip column = mean over emulsion pixels with that
    u-bin. Returns (Z, n_u_bins) array in [0, 1] intensity range.

    image_volume: (Z, H, W) float in [0, 1]
    u_map:        (H, W) float in [0, 1] (NaN outside film), z-invariant
    seg_2d:       (H, W) uint8 (0/1/2) — use class==2 for emulsion
    """
    Z, H, W = image_volume.shape
    valid = (seg_2d == 2) & np.isfinite(u_map)
    if not valid.any():
        raise ValueError("No emulsion pixels found in seg")
    u_at_valid = u_map[valid].astype(np.float64)
    u_idx = np.clip(
        (u_at_valid * (n_u_bins - 1)).astype(np.int32), 0, n_u_bins - 1
    )
    counts = np.bincount(u_idx, minlength=n_u_bins).astype(np.float32)
    rendered = np.zeros((Z, n_u_bins), dtype=np.float32)
    for z in range(Z):
        intensities = image_volume[z][valid].astype(np.float64)
        rendered[z] = np.bincount(
            u_idx, weights=intensities, minlength=n_u_bins
        ).astype(np.float32)
    nz_mask = counts > 0
    rendered[:, nz_mask] /= counts[nz_mask]
    # Fill empty bins by linear interpolation along u
    for z in range(Z):
        valid_u = nz_mask
        if valid_u.any() and (~valid_u).any():
            xs = np.where(valid_u)[0].astype(np.float64)
            ys = rendered[z, valid_u]
            x_all = np.arange(n_u_bins, dtype=np.float64)
            rendered[z] = np.interp(x_all, xs, ys).astype(np.float32)
    return rendered


def load_strip_for_training(args):
    """Build the training "strip" the CNN will learn to recognize.

    Modes:
      gt_strip          : direct ground_truth.npz["strip"] (source frames in [0, 1])
      rendered_at_gt    : sample CT image at GT u-map positions — matches the
                          actual distribution the CNN sees during unwrapping
                          (CT-intensity values, bilinear-softened frame edges).
    """
    if args.strip_mode == "gt_strip":
        print(f"Loading GT strip: {args.gt_npz}")
        gt = np.load(args.gt_npz)
        if "strip" not in gt.files:
            raise KeyError(f"{args.gt_npz} has no 'strip' key; found {gt.files}")
        return gt["strip"].astype(np.float32)
    elif args.strip_mode == "rendered_at_gt":
        import h5py
        print(f"Loading volume: {args.volume_h5}")
        with h5py.File(args.volume_h5, "r") as f:
            vol = f["volume"][:]  # (Z, H, W) uint16 or float
        # Normalize to [0, 1] if uint
        if vol.dtype != np.float32:
            vol = vol.astype(np.float32) / float(np.iinfo(vol.dtype).max)
        # Load probabilities (segmentation)
        print(f"Loading probs: {args.probs_h5}")
        with h5py.File(args.probs_h5, "r") as f:
            probs = f["exported_data"][:]  # (Z, H, W, 3)
        seg_3d = np.argmax(probs, axis=-1).astype(np.uint8)
        seg_2d = seg_3d[seg_3d.shape[0] // 2]
        print(f"Loading GT u-map: {args.gt_npz}")
        gt = np.load(args.gt_npz)
        u_map = gt["u_map"].astype(np.float32)
        print(f"Rendering strip from GT (n_u_bins={args.n_u_bins})...")
        return render_strip_from_gt(vol, u_map, seg_2d, args.n_u_bins)
    else:
        raise ValueError(f"Unknown strip_mode: {args.strip_mode}")


def train_discriminator(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strip = load_strip_for_training(args)
    print(f"  strip shape: {strip.shape}, range [{strip.min():.3f}, {strip.max():.3f}]")
    n_z = strip.shape[0]

    rng = np.random.RandomState(args.seed)
    model = StripDiscriminator(n_z=n_z, patch_w=args.patch_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  StripDiscriminator: {n_params:,} params, patch_w={args.patch_w}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    bce = nn.BCEWithLogitsLoss()

    log = []
    t0 = time.time()
    print(f"  starting training loop, {args.steps} steps", flush=True)
    for step in range(1, args.steps + 1):
        t_batch = time.time()
        x_np, y_np = build_batch(
            strip, args.batch_size, args.patch_w, rng=rng, batch_idx=step,
        )
        t_batch = time.time() - t_batch
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        _, logits = model(x)
        loss = bce(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == y).float().mean().item()
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  loss={loss.item():.4f}  acc={acc:.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.5f}  "
                  f"batch_t={t_batch*1000:.0f}ms  ({elapsed:.0f}s)",
                  flush=True)
            log.append({"step": step, "loss": float(loss.item()),
                        "acc": float(acc)})

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "strip_discriminator.pt")
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "n_z": n_z,
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")

    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)


def load_discriminator(ckpt_path, device="cuda"):
    """Load a trained StripDiscriminator from checkpoint. Frozen for inference."""
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    n_z = ckpt["n_z"]
    model = StripDiscriminator(
        n_z=n_z, patch_w=a["patch_w"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


# ─── CLI ──────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-npz", required=True,
                   help="Path to ground_truth.npz (needs 'strip' for "
                        "gt_strip mode, 'u_map' for rendered_at_gt).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--strip-mode", choices=["gt_strip", "rendered_at_gt"],
                   default="rendered_at_gt",
                   help="gt_strip: train on source frames in [0,1]. "
                        "rendered_at_gt: render strip from CT image at GT "
                        "u-map positions — matches the distribution the CNN "
                        "sees during unwrapping training.")
    p.add_argument("--volume-h5", default=None,
                   help="(rendered_at_gt mode) volume HDF5 path.")
    p.add_argument("--probs-h5", default=None,
                   help="(rendered_at_gt mode) probabilities HDF5 path.")
    p.add_argument("--n-u-bins", type=int, default=4096,
                   help="(rendered_at_gt mode) u-resolution of rendered strip.")
    p.add_argument("--patch-w", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_discriminator(args)


if __name__ == "__main__":
    main()
