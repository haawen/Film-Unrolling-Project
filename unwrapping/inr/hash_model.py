"""
H1 — Instant-NGP-style multi-resolution hash encoding (pure PyTorch).

Müller et al., SIGGRAPH 2022. Stack of L multi-resolution feature tables,
each of size T entries × F features. At each level the input coordinate is
mapped to a virtual grid; the corners of the enclosing cell are hashed
(modulo T) into the table and bilinearly interpolated. Levels are
concatenated and fed to a small MLP head.

For 2-D `(u, z)` input this is the same idea as `grid_model.GridDeformation`
but the per-level storage is capped at 2**log2_hashmap_size entries, with
collisions disambiguated by the MLP. Kept as a reference baseline because
in 2-D the dense version (grid_model) is usually strictly better — see
`.claude/docs/ss_unwrapping_plan.md`.

Same `forward(coords) -> (B, 2)` interface as `DeformationINR`. Zero-init
residual at step 0 via zero-init MLP head's final linear layer.

The pure-PyTorch implementation here is ~3× slower than `tinycudann` but
straightforward to read and free of CUDA/tcnn dependencies.
"""

import torch
import torch.nn as nn

# Prime numbers used by Instant-NGP for spatial hashing. Index by dimension.
_PRIMES_2D = (1, 2654435761)


def _hash_2d(coords_int: torch.Tensor, table_size: int) -> torch.Tensor:
    """XOR-prime spatial hash (Instant-NGP, eq. 4) for 2-D integer coords.

    coords_int: (..., 2) integer tensor.
    table_size: T (number of entries in the hash table).
    returns:    (...,) long indices in [0, table_size).
    """
    h = coords_int[..., 0] * _PRIMES_2D[0]
    h = h ^ (coords_int[..., 1] * _PRIMES_2D[1])
    return h.long() % table_size


class HashGridEncoding(nn.Module):
    """Multi-resolution 2-D hash-grid encoding."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 8192,
        table_init_std: float = 1e-4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.table_size = 2 ** log2_hashmap_size

        # Per-level resolution: geometric progression.
        b = (finest_resolution / base_resolution) ** (1.0 / max(1, n_levels - 1))
        resolutions = [int(round(base_resolution * (b ** i))) for i in range(n_levels)]
        self.register_buffer(
            "resolutions",
            torch.tensor(resolutions, dtype=torch.long),
            persistent=False,
        )

        # One hash table per level. Below the dense threshold (T >= res²) we
        # could store dense, but for simplicity always use the hash; the
        # collision rate at small resolutions is ≈0 anyway.
        self.tables = nn.ParameterList()
        for _ in range(n_levels):
            t = torch.randn(self.table_size, n_features_per_level) * table_init_std
            self.tables.append(nn.Parameter(t))

    @property
    def out_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, 2) in [-1, 1]. returns: (B, n_levels * F)."""
        # Map [-1, 1] → [0, 1].
        c = (coords + 1.0) * 0.5
        c = c.clamp(0.0, 1.0)

        feats_all = []
        for level, table in enumerate(self.tables):
            res = int(self.resolutions[level].item())
            scaled = c * res                           # (B, 2)
            lo = torch.floor(scaled).long()            # (B, 2)
            frac = scaled - lo.float()                 # (B, 2) in [0, 1]
            hi = lo + 1

            # Four corners of the enclosing cell.
            c00 = torch.stack([lo[:, 0], lo[:, 1]], dim=-1)
            c10 = torch.stack([hi[:, 0], lo[:, 1]], dim=-1)
            c01 = torch.stack([lo[:, 0], hi[:, 1]], dim=-1)
            c11 = torch.stack([hi[:, 0], hi[:, 1]], dim=-1)

            i00 = _hash_2d(c00, self.table_size)
            i10 = _hash_2d(c10, self.table_size)
            i01 = _hash_2d(c01, self.table_size)
            i11 = _hash_2d(c11, self.table_size)

            f00 = table[i00]
            f10 = table[i10]
            f01 = table[i01]
            f11 = table[i11]

            fu = frac[:, 0:1]
            fv = frac[:, 1:2]
            f0 = f00 * (1.0 - fu) + f10 * fu
            f1 = f01 * (1.0 - fu) + f11 * fu
            feat = f0 * (1.0 - fv) + f1 * fv          # (B, F)
            feats_all.append(feat)

        return torch.cat(feats_all, dim=-1)


class HashDeformation(nn.Module):
    """Hash encoding + small MLP → (dx, dy)."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 8192,
        hidden_dim: int = 64,
        n_layers: int = 2,
        output_dim: int = 2,
    ):
        super().__init__()
        self.encoding = HashGridEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )

        layers = []
        d_in = self.encoding.out_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d_in = hidden_dim
        head = nn.Linear(d_in, output_dim)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(coords))

    def param_groups(self, hash_lr: float, mlp_lr: float):
        return [
            {"params": list(self.encoding.parameters()), "lr": hash_lr},
            {"params": list(self.mlp.parameters()), "lr": mlp_lr},
        ]
