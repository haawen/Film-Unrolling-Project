"""
INR model architectures for CT volume representation.

Implements:
- FourierFeatureNetwork: random Fourier feature encoding + MLP
- SIREN: sinusoidal activation MLP (Sitzmann et al., 2020)

Both support optional segmentation head (3-class output alongside intensity).
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------

class FourierFeatureEncoding(nn.Module):
    """
    Random Fourier feature mapping: γ(x) = [cos(2π B x), sin(2π B x)]

    B is a fixed random matrix sampled from N(0, σ²).
    Output dimension = 2 * n_features.
    """

    def __init__(self, in_dim: int, n_features: int = 256, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, x):
        proj = 2 * math.pi * x @ self.B  # (batch, n_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# ---------------------------------------------------------------------------
# MLP Backbones
# ---------------------------------------------------------------------------

class ReLUMLP(nn.Module):
    """Standard ReLU MLP with optional skip connection at the midpoint."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_layers: int = 4,
                 out_dim: int = 1, skip_layer: int = None):
        super().__init__()
        self.skip_layer = skip_layer

        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            if i == skip_layer:
                d_in = hidden_dim + in_dim  # skip connection
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList()
        for i in range(0, len(layers), 2):
            self.layers.append(nn.Sequential(layers[i], layers[i + 1]))

        self.head = nn.Linear(hidden_dim, out_dim)
        self.in_dim = in_dim

    def forward(self, x, return_features=False):
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
        if return_features:
            return h
        return self.head(h)


class SirenLayer(nn.Module):
    """Single SIREN layer: Linear + sin activation."""

    def __init__(self, in_dim: int, out_dim: int, is_first: bool = False,
                 omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)

        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                bound = math.sqrt(6 / in_dim) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """SIREN network: all-sinusoidal-activation MLP."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_layers: int = 4,
                 out_dim: int = 1, omega_0: float = 30.0):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(SirenLayer(d_in, hidden_dim, is_first=(i == 0),
                                     omega_0=omega_0))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

        # Final layer init
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / omega_0
            self.head.weight.uniform_(-bound, bound)

    def forward(self, x, return_features=False):
        h = self.layers(x)
        if return_features:
            return h
        return self.head(h)


# ---------------------------------------------------------------------------
# Full INR Models
# ---------------------------------------------------------------------------

class FourierFeatureINR(nn.Module):
    """
    Fourier Feature Network for volume representation.

    Maps coordinates → Fourier features → MLP → intensity (+ optional segmentation).
    """

    def __init__(self, coord_dim: int = 2, n_fourier: int = 256, sigma: float = 10.0,
                 hidden_dim: int = 256, n_layers: int = 4, use_seg_head: bool = False,
                 n_classes: int = 3):
        super().__init__()
        self.encoding = FourierFeatureEncoding(coord_dim, n_fourier, sigma)
        self.mlp = ReLUMLP(self.encoding.out_dim, hidden_dim, n_layers, out_dim=1,
                           skip_layer=n_layers // 2)
        self.use_seg_head = use_seg_head
        if use_seg_head:
            self.seg_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, coords):
        """
        Args:
            coords: (batch, coord_dim) in [-1, 1]
        Returns:
            dict with 'intensity' (batch, 1) and optionally 'seg_logits' (batch, 3)
        """
        feat = self.encoding(coords)
        intensity = self.mlp(feat)
        out = {"intensity": intensity}
        if self.use_seg_head:
            features = self.mlp(feat, return_features=True)
            out["seg_logits"] = self.seg_head(features)
        return out


class SirenINR(nn.Module):
    """
    SIREN for volume representation.

    Maps coordinates directly (no explicit encoding) → SIREN MLP → intensity (+ optional seg).
    """

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 256, n_layers: int = 5,
                 omega_0: float = 30.0, use_seg_head: bool = False, n_classes: int = 3):
        super().__init__()
        self.mlp = SirenMLP(coord_dim, hidden_dim, n_layers, out_dim=1,
                            omega_0=omega_0)
        self.use_seg_head = use_seg_head
        if use_seg_head:
            self.seg_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, coords):
        intensity = self.mlp(coords)
        out = {"intensity": intensity}
        if self.use_seg_head:
            features = self.mlp(coords, return_features=True)
            out["seg_logits"] = self.seg_head(features)
        return out


def build_model(arch: str, coord_dim: int = 2, use_seg_head: bool = False,
                **kwargs) -> nn.Module:
    """Factory function to build INR models."""
    if arch == "fourier":
        return FourierFeatureINR(
            coord_dim=coord_dim,
            n_fourier=kwargs.get("n_fourier", 256),
            sigma=kwargs.get("sigma", 10.0),
            hidden_dim=kwargs.get("hidden_dim", 256),
            n_layers=kwargs.get("n_layers", 4),
            use_seg_head=use_seg_head,
        )
    elif arch == "siren":
        return SirenINR(
            coord_dim=coord_dim,
            hidden_dim=kwargs.get("hidden_dim", 256),
            n_layers=kwargs.get("n_layers", 5),
            omega_0=kwargs.get("omega_0", 30.0),
            use_seg_head=use_seg_head,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
