"""
G1 — Dense multi-resolution 2D feature grid (TensoRF / K-Planes at d=2,
equivalently "Instant-NGP without the hash").

Stack of dense `(C, Z_l, U_l)` feature planes at increasing (u, z) resolution,
bilinearly sampled at the input coordinate and concatenated, then fed to a
small MLP head that emits the (dx, dy) deformation residual.

Same `forward(coords) -> (B, 2)` interface as `DeformationINR`, with coords in
`[-1, 1]^2`. Step-0 residual is zero (the MLP head's final linear layer is
zero-init).

Architecture rationale lives in `.claude/docs/ss_unwrapping_plan.md`
(section "Architecture literature review — alternatives to hash").
"""

import torch
import torch.nn as nn


def _default_levels(n_levels, base_u, base_z, finest_u, finest_z):
    """Geometric progression from (base_u, base_z) to (finest_u, finest_z)."""
    if n_levels == 1:
        return [(finest_u, finest_z)]
    ru = (finest_u / base_u) ** (1.0 / (n_levels - 1))
    rz = (finest_z / base_z) ** (1.0 / (n_levels - 1))
    levels = []
    for i in range(n_levels):
        u_l = max(2, int(round(base_u * (ru ** i))))
        z_l = max(2, int(round(base_z * (rz ** i))))
        levels.append((u_l, z_l))
    return levels


class GridDeformation(nn.Module):
    """Dense multi-resolution 2D feature grid + small MLP head.

    coords: (B, 2) in [-1, 1]; column 0 = u_norm, column 1 = z_norm.

    Convention matches `DeformationINR.forward(coords)`. The feature grid
    is stored with shape (1, C, Z_l, U_l) so `F.grid_sample` indexes the
    last two dims as (y=z, x=u) — matching coords where x=u, y=z.
    """

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        base_resolution_u: int = 64,
        base_resolution_z: int = 16,
        finest_resolution_u: int = 8192,
        finest_resolution_z: int = 256,
        hidden_dim: int = 64,
        n_layers: int = 2,
        output_dim: int = 2,
        grid_init_std: float = 1e-4,
    ):
        super().__init__()

        self.levels = _default_levels(
            n_levels, base_resolution_u, base_resolution_z,
            finest_resolution_u, finest_resolution_z,
        )
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        enc_dim = n_levels * n_features_per_level

        # One dense feature plane per level. Shape (C, Z_l, U_l).
        # Stored without a leading batch dim because we sample via manual
        # indexing rather than F.grid_sample — the latter lacks a 2nd-order
        # derivative implementation in PyTorch, which breaks the conformal
        # loss (which autodiffs through ∂pred_xy/∂uv) in SS mode.
        self.grids = nn.ParameterList()
        for (u_l, z_l) in self.levels:
            g = torch.randn(n_features_per_level, z_l, u_l) * grid_init_std
            self.grids.append(nn.Parameter(g))

        # MLP head: ReLU, deliberately small. Final layer zero-init.
        layers = []
        d_in = enc_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d_in = hidden_dim
        head = nn.Linear(d_in, output_dim)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Sample all feature planes at `coords` and concatenate.

        coords: (B, 2) in [-1, 1] with (u_norm, z_norm).
        returns: (B, n_levels * n_features_per_level)

        Manual bilinear interpolation via index-then-blend. This is fully
        differentiable wrt both the table values and the input coords
        (used by the SS conformal loss).
        """
        # Map [-1, 1] → [0, 1].
        c = (coords + 1.0) * 0.5
        # Clamp to [0, 1] so the floor/ceil never go out of bounds.
        c = c.clamp(0.0, 1.0)
        u_n, z_n = c[:, 0], c[:, 1]

        feats = []
        for g in self.grids:
            C, Z_l, U_l = g.shape
            # Pixel coords in [0, U_l-1] and [0, Z_l-1].
            u = u_n * (U_l - 1)
            z = z_n * (Z_l - 1)
            u0 = u.floor().long().clamp(0, U_l - 1)
            z0 = z.floor().long().clamp(0, Z_l - 1)
            u1 = (u0 + 1).clamp(0, U_l - 1)
            z1 = (z0 + 1).clamp(0, Z_l - 1)
            du = (u - u0.float()).unsqueeze(-1)   # (B, 1)
            dz = (z - z0.float()).unsqueeze(-1)   # (B, 1)

            # Gather 4 corners; result shape (B, C) each (we permute g[:, z, u]
            # which is (C, B) → transpose).
            f00 = g[:, z0, u0].t()
            f10 = g[:, z0, u1].t()
            f01 = g[:, z1, u0].t()
            f11 = g[:, z1, u1].t()

            f0 = f00 * (1.0 - du) + f10 * du
            f1 = f01 * (1.0 - du) + f11 * du
            feat = f0 * (1.0 - dz) + f1 * dz       # (B, C)
            feats.append(feat)
        return torch.cat(feats, dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encode(coords))

    # Used by surface_train.py to give the grid params a higher LR than the MLP.
    def param_groups(self, grid_lr: float, mlp_lr: float):
        return [
            {"params": list(self.grids.parameters()), "lr": grid_lr},
            {"params": list(self.mlp.parameters()), "lr": mlp_lr},
        ]
