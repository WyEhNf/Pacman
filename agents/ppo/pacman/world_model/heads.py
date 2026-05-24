# pacman/world_model/heads.py
"""Reward and continue prediction heads for the Pac-Man world model."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> nn.Sequential:
    """Build an MLP with SiLU activations between layers, no activation on output.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_layers: Total number of linear layers (including input and output).

    Returns:
        nn.Sequential MLP.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    layers: list[nn.Module] = []

    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        # Final layer: hidden_dim -> output_dim (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class RewardHead(nn.Module):
    """MLP that predicts reward from a latent state vector.

    Args:
        latent_dim: Dimension of the input latent vector.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of linear layers in the MLP.
    """

    def __init__(
        self,
        latent_dim: int = 2560,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.mlp = build_mlp(latent_dim, hidden_dim, 1, num_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state.

        Args:
            latent: Latent tensor of shape (B, latent_dim).

        Returns:
            Reward predictions of shape (B, 1).
        """
        return self.mlp(latent)


class ContinueHead(nn.Module):
    """MLP that predicts episode continuation logits from a latent state vector.

    Outputs raw logits; apply sigmoid externally to get probabilities.

    Args:
        latent_dim: Dimension of the input latent vector.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of linear layers in the MLP.
    """

    def __init__(
        self,
        latent_dim: int = 2560,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.mlp = build_mlp(latent_dim, hidden_dim, 1, num_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict continuation logits from latent state.

        Args:
            latent: Latent tensor of shape (B, latent_dim).

        Returns:
            Continuation logits of shape (B, 1). Apply sigmoid externally.
        """
        return self.mlp(latent)
