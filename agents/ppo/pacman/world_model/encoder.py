# pacman/world_model/encoder.py
"""CNN observation encoder for the Pac-Man world model."""
from __future__ import annotations

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encode an 8-channel grid observation + scalar features into a flat latent vector.

    Architecture:
    - Sequential CNN layers (Conv2d + SiLU) with padding=k//2
    - Scalar branch: Linear(num_scalars, 64) + SiLU
    - Fusion: Linear(cnn_flat + 64, output_dim) + SiLU

    Args:
        grid_channels: Number of input grid channels (default 8).
        num_scalars: Number of scalar features (default 5).
        cnn_channels: Output channels per CNN layer.
        cnn_kernels: Kernel size per CNN layer.
        cnn_strides: Stride per CNN layer.
        output_dim: Dimension of the output latent vector.
    """

    def __init__(
        self,
        grid_channels: int = 8,
        num_scalars: int = 5,
        cnn_channels: list[int] = (64, 128, 256, 256),
        cnn_kernels: list[int] = (4, 4, 4, 4),
        cnn_strides: list[int] = (2, 2, 2, 1),
        output_dim: int = 512,
    ) -> None:
        super().__init__()

        # Build CNN layers
        layers: list[nn.Module] = []
        in_ch = grid_channels
        for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2))
            layers.append(nn.SiLU())
            in_ch = out_ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Dynamically compute CNN flat size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, 31, 28)
            cnn_flat = self.cnn(dummy).shape[1]

        # Scalar embedding branch
        self.scalar_embed = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.SiLU(),
        )

        # Fusion linear + activation
        self.fusion = nn.Sequential(
            nn.Linear(cnn_flat + 64, output_dim),
            nn.SiLU(),
        )

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Encode observation into a latent vector.

        Args:
            grid: Grid observation of shape (B, grid_channels, 31, 28).
            scalars: Scalar features of shape (B, num_scalars).

        Returns:
            Latent tensor of shape (B, output_dim).
        """
        cnn_out = self.cnn(grid)                 # (B, cnn_flat)
        scalar_out = self.scalar_embed(scalars)  # (B, 64)
        combined = torch.cat([cnn_out, scalar_out], dim=1)
        return self.fusion(combined)             # (B, output_dim)
