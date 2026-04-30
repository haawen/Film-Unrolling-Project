"""
Per-winding parametric deformation model.

Drop-in replacement for the INR (`unwrap_model.DeformationINR`) that represents
the residual `(dx, dy)(u, z)` as a sum of independent per-winding Fourier
series in the local angle θ_local = 2π · (u - floor(u)).

Why: the shared-weight INR smears per-winding jitter via averaging across
windings. With ~28 windings × M harmonics × 4 (sin/cos × dx/dy) ≈ 1k params,
each winding has its own dedicated Fourier expansion — no shared weights to
interfere across windings.

Architecture:
    input  : (u_raw_unscaled, z_norm)  — same `uv` tensor as the INR for
             drop-in compatibility. We extract u_raw via the inverse of
             u_norm = u_raw / n_layers · 2 − 1.
    output : (dx, dy) residual in the same scale as the INR's output

For each sample:
    k         = floor(u_raw)             — winding index
    θ_local   = 2π · (u_raw − k)         — local angle within winding [0, 2π)
    dx        = Σ_{m=1..M} a_km cos(mθ) + b_km sin(mθ)
    dy        = Σ_{m=1..M} c_km cos(mθ) + d_km sin(mθ)

Parameters: shape (n_layers, 4, M) — packed as [a, b, c, d] for each winding.

Z-dependence: NOT included by default — synthetic geometry is z-invariant by
construction. Real Mickey may need z, but we'll add it only if needed.
Initialize all params to zero so step-0 prediction == analytical_xy (matches
INR's zero-init head).
"""

import math
import torch
import torch.nn as nn


class PerWindingParametric(nn.Module):
    """Per-winding Fourier-in-θ deformation residual.

    Output is unbounded; same scale conventions as `DeformationINR`.

    Args:
        n_layers:    number of windings detected in the data
        n_harmonics: M, harmonics per winding per dimension (default 10)
        z_dependent: if True, add a per-z low-rank correction
                     (n_z, n_layers, 4, M) on top — defaults to False
        n_z:         only used if z_dependent=True
    """

    def __init__(self, n_layers, n_harmonics=10, output_dim=2,
                 z_dependent=False, n_z=1):
        super().__init__()
        assert output_dim == 2, "PerWindingParametric only supports 2D output"
        self.n_layers = int(n_layers)
        self.M = int(n_harmonics)
        self.z_dependent = z_dependent

        # Coefficients: shape (n_layers, 4, M)
        # axes: winding k, [a, b, c, d], harmonic m
        self.coeffs = nn.Parameter(
            torch.zeros(self.n_layers, 4, self.M)
        )

        if z_dependent:
            # Per-z multiplicative correction (small): (n_z, n_layers, 4, M)
            self.coeffs_z = nn.Parameter(
                torch.zeros(int(n_z), self.n_layers, 4, self.M)
            )
        else:
            self.coeffs_z = None

        # Precompute m-indices for the Fourier expansion: (M,)
        self.register_buffer("m_idx", torch.arange(1, self.M + 1).float())

    def _u_raw_from_uv(self, uv):
        """Recover u_raw ∈ [0, n_layers) from uv ∈ [-1, 1]² where
        uv[:, 0] = u_norm = u_raw / n_layers · 2 − 1.
        """
        u_norm = uv[:, 0]
        u_raw = (u_norm + 1.0) * 0.5 * self.n_layers
        return u_raw.clamp(0.0, self.n_layers - 1e-6)

    def _z_idx_from_uv(self, uv):
        """Recover z_idx (long) from uv[:, 1] = z_norm ∈ [-1, 1] for n_z slices.
        Only used in z_dependent mode.
        """
        z_norm = uv[:, 1]
        # We don't have n_z directly here; use coeffs_z's first dim.
        n_z = self.coeffs_z.shape[0] if self.coeffs_z is not None else 1
        if n_z <= 1:
            return torch.zeros(uv.shape[0], dtype=torch.long, device=uv.device)
        z_idx = ((z_norm + 1.0) * 0.5 * (n_z - 1)).round().long()
        return z_idx.clamp(0, n_z - 1)

    def forward(self, uv):
        """
        Args:
            uv: (B, 2) — uv[:, 0] = u_norm in [-1, 1], uv[:, 1] = z_norm

        Returns:
            xy_res: (B, 2) — residual (dx, dy) in [-1, 1] normalized coords
        """
        B = uv.shape[0]
        u_raw = self._u_raw_from_uv(uv)              # (B,)
        k = u_raw.long()                             # (B,) winding index
        theta = 2.0 * math.pi * (u_raw - k.float())  # (B,) local angle

        # Per-sample Fourier basis: cos(mθ), sin(mθ) for m=1..M
        # m_idx: (M,), theta: (B,) → (B, M)
        m_theta = theta.unsqueeze(1) * self.m_idx.unsqueeze(0)  # (B, M)
        cos_m = torch.cos(m_theta)                              # (B, M)
        sin_m = torch.sin(m_theta)                              # (B, M)

        # Gather per-sample coeffs: (B, 4, M) from (n_layers, 4, M)[k]
        c = self.coeffs[k]                                       # (B, 4, M)

        if self.z_dependent and self.coeffs_z is not None:
            z_idx = self._z_idx_from_uv(uv)                      # (B,)
            cz = self.coeffs_z[z_idx, k]                         # (B, 4, M)
            c = c + cz

        # dx = Σ_m a_m cos(mθ) + b_m sin(mθ)
        # dy = Σ_m c_m cos(mθ) + d_m sin(mθ)
        a = c[:, 0]                                              # (B, M)
        b = c[:, 1]                                              # (B, M)
        c_y = c[:, 2]                                            # (B, M)
        d_y = c[:, 3]                                            # (B, M)
        dx = (a * cos_m + b * sin_m).sum(dim=1)                  # (B,)
        dy = (c_y * cos_m + d_y * sin_m).sum(dim=1)              # (B,)

        return torch.stack([dx, dy], dim=1)                      # (B, 2)

    def num_params(self):
        n = self.coeffs.numel()
        if self.coeffs_z is not None:
            n += self.coeffs_z.numel()
        return n
