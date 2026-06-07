"""
Cubic B-spline tensor-product deformation model (Track 3).

Drop-in replacement for `DeformationINR`: same `forward(uv) -> (dx, dy)`
interface. Maps (u_norm, z_norm) in [-1, 1]² to a residual offset in
normalized image coords. Used by `surface_train.py` when
`--model-type bspline` is set.

Architecture:
- Open-uniform cubic B-spline basis, separately along u and z.
- Tensor-product: each output coordinate is `sum_ij b_u[i] * b_z[j] * P[i,j]`.
- Control points are the only learnable parameters: shape
  `(n_u_ctrl, n_z_ctrl, output_dim)` where n_*_ctrl = n_*_spans + 3.

Why B-spline:
- The previous attempt (`PerWindingParametric`) used independent per-winding
  Fourier-in-θ series with no continuity coupling at integer u — C⁰
  discontinuity at every winding boundary collapsed row_corr to 0.21.
- Cubic B-splines guarantee C² continuity at knots, so per-winding control
  is preserved without seam jumps.
- A small number of control points (typically 1k–4k) means no shared MLP
  weights blurring across windings — different from the INR's smoothness
  prior that capped the supervised result at 0.91 on video imperfect.

Default sizing:
- `n_u_spans = 4 * n_windings`: four control points per winding gives enough
  expressivity for the per-winding sinusoidal jitter the synthetic generator
  injects (≤3 components × per-winding phase shifts).
- `n_z_spans = 3`: synthetic geometry is z-invariant; only a few z DoF needed.
  More may help on real Mickey where geometry varies slightly with z.
"""

import torch
import torch.nn as nn


def _cubic_b_spline_basis(f):
    """Cubic B-spline basis at fractional parameter f ∈ [0, 1).

    Returns the four nonzero basis weights for indices [-1, 0, 1, 2] relative
    to the active span — the standard uniform-cubic-B-spline coefficients.

    Args:
        f: (B,) tensor in [0, 1).
    Returns:
        (B, 4) basis weights.
    """
    one_minus_f = 1.0 - f
    b0 = (one_minus_f ** 3) / 6.0
    b1 = (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) / 6.0
    b2 = (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) / 6.0
    b3 = (f ** 3) / 6.0
    return torch.stack([b0, b1, b2, b3], dim=-1)


class BSplineDeformation(nn.Module):
    """2D tensor-product cubic B-spline residual map (u, z) → (dx, dy)."""

    def __init__(self, n_u_spans, n_z_spans=3, output_dim=2):
        super().__init__()
        assert n_u_spans >= 1 and n_z_spans >= 1
        self.n_u_spans = int(n_u_spans)
        self.n_z_spans = int(n_z_spans)
        self.output_dim = int(output_dim)
        # Need n_spans + 3 control points along each axis for cubic basis
        # to cover [0, n_spans] with 4-active-knot windows.
        self.n_u_ctrl = self.n_u_spans + 3
        self.n_z_ctrl = self.n_z_spans + 3
        # Zero-init: step-0 prediction == analytical_xy (no residual).
        self.control = nn.Parameter(
            torch.zeros(self.n_u_ctrl, self.n_z_ctrl, self.output_dim)
        )

    def forward(self, uv):
        """
        Args:
            uv: (B, 2) tensor with `uv[:, 0]` = u_norm in [-1, 1],
                `uv[:, 1]` = z_norm in [-1, 1].
        Returns:
            (B, output_dim) residual offset.
        """
        B = uv.shape[0]
        # Map [-1, 1] → [0, n_spans].
        tu = (uv[:, 0] + 1.0) * 0.5 * self.n_u_spans
        tz = (uv[:, 1] + 1.0) * 0.5 * self.n_z_spans

        iu = torch.clamp(torch.floor(tu).long(), 0, self.n_u_spans - 1)
        iz = torch.clamp(torch.floor(tz).long(), 0, self.n_z_spans - 1)
        fu = (tu - iu.to(tu.dtype)).clamp(0.0, 1.0)
        fz = (tz - iz.to(tz.dtype)).clamp(0.0, 1.0)

        bu = _cubic_b_spline_basis(fu)  # (B, 4)
        bz = _cubic_b_spline_basis(fz)  # (B, 4)

        # Outer product per-sample → (B, 4, 4) basis weights.
        weights = bu.unsqueeze(2) * bz.unsqueeze(1)  # (B, 4, 4)

        # Index control points: (B, 4, 4, output_dim).
        # iu spans 0..n_u_spans-1 → control rows iu..iu+3 (within bounds since
        # n_u_ctrl = n_u_spans + 3).
        offsets = torch.arange(4, device=uv.device)
        u_idx = iu.unsqueeze(1) + offsets.unsqueeze(0)  # (B, 4)
        z_idx = iz.unsqueeze(1) + offsets.unsqueeze(0)  # (B, 4)
        # Gather: (B, 4_u, 4_z, output_dim).
        pts = self.control[u_idx.unsqueeze(2).expand(-1, -1, 4),
                           z_idx.unsqueeze(1).expand(-1, 4, -1)]
        # Contract basis weights against control points.
        out = (weights.unsqueeze(-1) * pts).sum(dim=(1, 2))  # (B, output_dim)
        return out
