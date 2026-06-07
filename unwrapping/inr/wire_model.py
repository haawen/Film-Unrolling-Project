"""
S2 — WIRE: Wavelet Implicit Neural Representations (Saragadam et al., CVPR 2023).

Replaces ReLU/sine activations with the real-valued Gabor wavelet
    psi(x) = sin(omega * x) * exp(-(sigma * x)^2)
which gives each neuron both frequency selectivity (the sine) and spatial
compactness (the Gaussian envelope). Empirically reduces ringing artefacts
relative to SIREN and edge-blur relative to Gaussian-only activations.

Note: the official WIRE uses complex Gabor wavelets internally; we use the
real-valued form (Saragadam et al., supplementary § "Real-valued WIRE")
for simplicity and to match the (B, 2) real-valued output our pipeline
expects. The first layer's weights and biases are sampled from a wider
uniform per the paper.

Same `forward(coords) -> (B, 2)` interface as `DeformationINR`. Zero-init
residual at step 0 via zero-init final linear layer.
"""

import math

import torch
import torch.nn as nn


class GaborLayer(nn.Module):
    """Linear → sin(ω·x) * exp(-(σ·x)^2). Acts neuron-wise on the pre-activation."""

    def __init__(self, in_dim: int, out_dim: int, omega: float, sigma: float,
                 is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        self.sigma = sigma
        self._init_weights(is_first)

    def _init_weights(self, is_first: bool):
        # Per the WIRE paper: first-layer weights span a wider range so the
        # Gabor activation explores a useful frequency band immediately.
        with torch.no_grad():
            if is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / max(self.omega, 1e-3)
            self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return torch.sin(self.omega * h) * torch.exp(-(self.sigma * h) ** 2)


class WIREDeformation(nn.Module):
    """WIRE Gabor-wavelet MLP that maps (u, z) → (dx, dy)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        omega: float = 20.0,
        sigma: float = 10.0,
        input_dim: int = 2,
        output_dim: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        d_in = input_dim
        for i in range(n_layers):
            self.layers.append(
                GaborLayer(d_in, hidden_dim, omega=omega, sigma=sigma,
                           is_first=(i == 0))
            )
            d_in = hidden_dim

        # Final linear head, zero-init for residual-on-analytical convention.
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = coords
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
