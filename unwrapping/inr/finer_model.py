"""
S3 — FINER: Flexible spectral-bias tuning in INR by variable-periodic
activation functions (Liu et al., CVPR 2024).

Drop-in upgrade for SIREN. The activation is
    phi(x) = sin(omega_0 * (|x| + 1) * x)
which is "variable-periodic" — the effective frequency depends on the
magnitude of the pre-activation. Initialising the per-neuron bias from a
*wider* uniform U(-k, k), k >= 1, pushes pre-activations into the high-|x|
regime where phi has more high-frequency content; this widens the
network's reachable spectral support beyond standard SIREN.

Weight init follows SIREN (Sitzmann et al. 2020): first layer
U(-1/in_dim, 1/in_dim), subsequent layers U(-sqrt(6/in_dim)/omega_0, ...).

Same `forward(coords) -> (B, 2)` interface as `DeformationINR`. Zero-init
residual at step 0 via zero-init final linear layer.
"""

import math

import torch
import torch.nn as nn


def _finer_activation(x: torch.Tensor, omega_0: float) -> torch.Tensor:
    return torch.sin(omega_0 * (x.abs() + 1.0) * x)


class FinerLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float,
                 is_first: bool = False, bias_k: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega_0 = omega_0
        self.is_first = is_first
        self._init_weights(bias_k)

    def _init_weights(self, bias_k: float):
        in_dim = self.linear.in_features
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / max(self.omega_0, 1e-3)
            self.linear.weight.uniform_(-bound, bound)
            # FINER's key change: wider bias init unlocks higher frequencies.
            self.linear.bias.uniform_(-bias_k, bias_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _finer_activation(self.linear(x), self.omega_0)


class FINERDeformation(nn.Module):
    """FINER variable-periodic MLP that maps (u, z) → (dx, dy)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        omega_0: float = 30.0,
        bias_k: float = 5.0,
        input_dim: int = 2,
        output_dim: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        d_in = input_dim
        for i in range(n_layers):
            self.layers.append(
                FinerLayer(d_in, hidden_dim, omega_0=omega_0,
                           is_first=(i == 0), bias_k=bias_k)
            )
            d_in = hidden_dim

        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = coords
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
