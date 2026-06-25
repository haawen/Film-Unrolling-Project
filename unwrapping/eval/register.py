"""Cross-modal spatial registration for the film-video comparison.

FFT phase-correlation (the first scaffold) locks onto the smooth shading /
background on low-contrast cartoon frames and fails (only ~19% of pairs aligned
<25px). This module registers on **gradient-magnitude maps** (edge structure is
modality-invariant and suppresses the smooth shading) with a **similarity
transform** (translation + isotropic scale + rotation) optimized by torch
autograd against **local normalized cross-correlation (LNCC)** — robust to the
CT-vs-optical intensity gap and to the small scale/rotation left after the
discrete rot90/flip + picture-area crop. No cv2/skimage needed.

Public API:
    register(fixed, moving, ...) -> (warped_moving, info)
        fixed, moving : 2D float32 [0,1]; returns moving warped onto fixed.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _grad_mag(t):
    """Sobel gradient magnitude of (1,1,H,W) tensor, per-image normalized."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=t.dtype, device=t.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    g = g - g.mean()
    s = g.std()
    return g / (s + 1e-6)


def _lncc(a, b, win=9):
    """Mean local normalized cross-correlation of (1,1,H,W) maps."""
    pad = win // 2
    pool = lambda x: F.avg_pool2d(x, win, stride=1, padding=pad,
                                  count_include_pad=False)
    mu_a, mu_b = pool(a), pool(b)
    va = pool(a * a) - mu_a * mu_a
    vb = pool(b * b) - mu_b * mu_b
    vab = pool(a * b) - mu_a * mu_b
    cc = vab / torch.sqrt(torch.clamp(va * vb, min=1e-6))
    return cc.mean()


def _theta(params):
    """params (tx, ty, log_s, rot) -> (1,2,3) affine matrix (normalized coords)."""
    tx, ty, log_s, rot = params
    s = torch.exp(log_s)
    c, sn = torch.cos(rot), torch.sin(rot)
    return torch.stack([
        torch.stack([s * c, -s * sn, tx]),
        torch.stack([s * sn, s * c, ty]),
    ]).unsqueeze(0)


def _warp(moving, params):
    grid = F.affine_grid(_theta(params), moving.shape, align_corners=False)
    return F.grid_sample(moving, grid, align_corners=False,
                         padding_mode="border")


def register(fixed, moving, iters=250, lr=3e-2, reg_size=160,
             win=9, restarts=(0.0, 0.06, -0.06)):
    """Register `moving` onto `fixed` (2D [0,1]). Returns (warped, info).

    Registration runs on downscaled gradient maps for speed; the recovered
    similarity transform is resolution-independent and re-applied at full size.
    A few rotation restarts guard against the LNCC local-minimum on near-blank
    cartoon frames. `info` has the final LNCC and transform params.
    """
    dev = "cpu"
    fx = torch.from_numpy(np.ascontiguousarray(fixed)).float()[None, None]
    mv = torch.from_numpy(np.ascontiguousarray(moving)).float()[None, None]

    # downscaled copies for the optimization
    h = reg_size
    wf = max(2, int(round(h * fx.shape[3] / fx.shape[2])))
    wm = max(2, int(round(h * mv.shape[3] / mv.shape[2])))
    fxs = _grad_mag(F.interpolate(fx, (h, wf), mode="bilinear",
                                  align_corners=False))
    mvs = _grad_mag(F.interpolate(mv, (h, wm), mode="bilinear",
                                  align_corners=False))

    best = None
    for r0 in restarts:
        p = [torch.zeros((), device=dev, requires_grad=True) for _ in range(3)]
        p.append(torch.tensor(float(r0), requires_grad=True))
        opt = torch.optim.Adam(p, lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            warped = _warp(mvs, p)
            # match warped-moving grid to fixed grid by resampling to fixed size
            if warped.shape[2:] != fxs.shape[2:]:
                warped = F.interpolate(warped, fxs.shape[2:], mode="bilinear",
                                       align_corners=False)
            loss = 1.0 - _lncc(warped, fxs, win)
            loss.backward()
            opt.step()
        with torch.no_grad():
            warped = _warp(mvs, p)
            if warped.shape[2:] != fxs.shape[2:]:
                warped = F.interpolate(warped, fxs.shape[2:], mode="bilinear",
                                       align_corners=False)
            score = float(_lncc(warped, fxs, win))
        if best is None or score > best[0]:
            best = (score, [q.detach().clone() for q in p])

    score, params = best
    with torch.no_grad():
        # apply to full-res moving, resampled onto fixed's grid
        full = _warp(mv, params)
        full = F.interpolate(full, fx.shape[2:], mode="bilinear",
                             align_corners=False)
    warped_np = full[0, 0].numpy().astype(np.float32)
    info = {
        "lncc": score,
        "tx": float(params[0]), "ty": float(params[1]),
        "scale": float(np.exp(float(params[2]))), "rot_deg": float(params[3]) * 180 / np.pi,
    }
    return warped_np, info
