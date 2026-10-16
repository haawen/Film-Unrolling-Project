"""
Deformation network for learned unwrapping (Step 1b).

Maps film pixel coordinates (x, y) → (u, v) on the unrolled strip.
Uses Fourier Feature encoding + ReLU MLP (same backbone as Step 1a).

Output is unbounded (no sigmoid) — targets are standardized to zero mean, unit variance.
"""

import math
import torch
import torch.nn as nn


class DeformationINR(nn.Module):
    """Fourier Feature Network that maps coordinates → (u, v).

    Architecture:
        (x, y[, z]) → FourierFeatures → MLP → (u, v)

    Supports 2D input (x, y) or 3D input (x, y, z).
    Output is raw (unbounded). Normalization is handled externally via
    target standardization (mean/std stored in the dataset).
    """

    def __init__(self, n_fourier=256, sigma=10.0, hidden_dim=256, n_layers=4,
                 input_dim=2, output_dim=2):
        super().__init__()
        self.input_dim = input_dim

        # Fourier feature encoding
        B = torch.randn(input_dim, n_fourier) * sigma
        self.register_buffer("B", B)
        enc_dim = 2 * n_fourier

        # MLP with skip connection at midpoint
        skip_layer = n_layers // 2
        layers = []
        for i in range(n_layers):
            d_in = enc_dim if i == 0 else hidden_dim
            if i == skip_layer:
                d_in = hidden_dim + enc_dim
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList()
        for i in range(0, len(layers), 2):
            self.layers.append(nn.Sequential(layers[i], layers[i + 1]))

        self.skip_layer = skip_layer

        # Output head: no activation, unbounded
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode(self, x):
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def forward(self, coords):
        """
        Args:
            coords: (batch, 2) pixel coordinates in [-1, 1]

        Returns:
            uv: (batch, 2) predicted (u, v) — raw, unbounded
        """
        enc = self.encode(coords)
        h = enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, enc], dim=-1)
            h = layer(h)
        return self.head(h)
