# pacman/world_model/decoder.py
"""Transposed-CNN observation decoder for the Pac-Man world model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationDecoder(nn.Module):
    """Decode a latent vector into a grid observation and scalar features.

    Architecture:
    - FC: Linear(latent_dim, start_channels * start_h * start_w) + SiLU, reshape to (B, C, H, W)
    - Transposed CNN layers: ConvTranspose2d + SiLU (all but last)
    - Final ConvTranspose2d outputs grid_channels with no activation
    - F.interpolate to guarantee exact (31, 28) output shape (MPS-compatible)
    - Separate scalar head: Linear(latent_dim, 256) + SiLU + Linear(256, num_scalars)

    Args:
        latent_dim: Dimension of the input latent vector.
        grid_channels: Number of output grid channels.
        num_scalars: Number of output scalar features.
        cnn_channels: Output channels per transposed CNN layer.
        cnn_kernels: Kernel size per transposed CNN layer.
        cnn_strides: Stride per transposed CNN layer.
    """

    _START_H = 4
    _START_W = 4

    def __init__(
        self,
        latent_dim: int = 2560,
        grid_channels: int = 8,
        num_scalars: int = 5,
        cnn_channels: list[int] = (256, 256, 128, 64),
        cnn_kernels: list[int] = (4, 4, 4, 4),
        cnn_strides: list[int] = (1, 2, 2, 2),
    ) -> None:
        super().__init__()

        start_channels = cnn_channels[0]

        # FC projection + reshape preparation
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, start_channels * self._START_H * self._START_W),
            nn.SiLU(),
        )
        self._start_channels = start_channels

        # Build transposed CNN layers
        deconv_layers: list[nn.Module] = []
        in_ch = start_channels
        channel_list = list(cnn_channels[1:]) + [grid_channels]
        for i, (out_ch, k, s) in enumerate(zip(channel_list, cnn_kernels, cnn_strides)):
            deconv_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)
            )
            if i < len(channel_list) - 1:
                # All but the last get SiLU
                deconv_layers.append(nn.SiLU())
            in_ch = out_ch

        self.deconv = nn.Sequential(*deconv_layers)

        # Scalar reconstruction head (independent of deconv path)
        self.scalar_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_scalars),
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent vector into grid and scalar observations.

        Args:
            latent: Latent tensor of shape (B, latent_dim).

        Returns:
            Tuple of:
            - grid: Reconstructed grid of shape (B, grid_channels, 31, 28).
            - scalars: Reconstructed scalars of shape (B, num_scalars).
        """
        B = latent.shape[0]

        # Grid branch
        x = self.fc(latent)  # (B, start_channels * H * W)
        x = x.view(B, self._start_channels, self._START_H, self._START_W)
        grid = self.deconv(x)
        # Resize to exact (31, 28) — bilinear interpolate is MPS-compatible
        if grid.shape[-2:] != (31, 28):
            grid = F.interpolate(grid, size=(31, 28), mode="bilinear", align_corners=False)

        # Scalar branch
        scalars = self.scalar_head(latent)  # (B, num_scalars)

        return grid, scalars
