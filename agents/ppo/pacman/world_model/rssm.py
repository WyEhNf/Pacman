# pacman/world_model/rssm.py
"""Recurrent State-Space Model (RSSM) dynamics for the Pac-Man world model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """RSSM dynamics model with categorical stochastic state and GRU deterministic state.

    The latent state has two parts:
    - Deterministic state h: GRU hidden state (gru_hidden dims) -- long-term memory
    - Stochastic state z: Categorical distribution, flattened
      (stoch_classes x stoch_categoricals dims) -- current situation

    Args:
        stoch_classes: Number of classes per categorical distribution.
        stoch_categoricals: Number of categorical distributions.
        gru_hidden: Dimension of the GRU hidden state.
        action_dim: Number of discrete actions.
        encoder_output_dim: Dimension of the encoder output vector.
    """

    def __init__(
        self,
        stoch_classes: int = 32,
        stoch_categoricals: int = 64,
        gru_hidden: int = 512,
        action_dim: int = 4,
        encoder_output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.stoch_classes = stoch_classes
        self.stoch_categoricals = stoch_categoricals
        self.stoch_dim = stoch_classes * stoch_categoricals
        self.gru_hidden = gru_hidden

        # 1. Action embedding
        self.action_embed = nn.Embedding(action_dim, 64)

        # 2. GRU pre-processing: Linear(stoch_dim + 64, gru_hidden) + SiLU
        self.gru_pre = nn.Sequential(
            nn.Linear(self.stoch_dim + 64, gru_hidden),
            nn.SiLU(),
        )

        # 3. GRU cell
        self.gru = nn.GRUCell(gru_hidden, gru_hidden)

        # 4. Prior network: predicts z from h alone (for imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(gru_hidden, 512),
            nn.SiLU(),
            nn.Linear(512, self.stoch_dim),
        )

        # 5. Posterior network: predicts z from h + encoder output (during training)
        self.posterior_net = nn.Sequential(
            nn.Linear(gru_hidden + encoder_output_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.stoch_dim),
        )

    @property
    def latent_dim(self) -> int:
        """Total latent dimension: gru_hidden + stoch_dim."""
        return self.gru_hidden + self.stoch_dim

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, z) on the model's device.

        Args:
            batch_size: Number of parallel environments / sequences.

        Returns:
            Tuple of (h, z) where h has shape (B, gru_hidden) and z has shape (B, stoch_dim).
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.gru_hidden, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def dynamics(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Advance the deterministic state by one step.

        Embeds the action, concatenates with z, passes through gru_pre, then GRU.

        Args:
            h: Current deterministic state of shape (B, gru_hidden).
            z: Current stochastic state of shape (B, stoch_dim).
            action: Discrete actions of shape (B,).

        Returns:
            Next deterministic state h_next of shape (B, gru_hidden).
        """
        action_emb = self.action_embed(action)          # (B, 64)
        x = torch.cat([z, action_emb], dim=1)           # (B, stoch_dim + 64)
        x = self.gru_pre(x)                             # (B, gru_hidden)
        h_next = self.gru(x, h)                         # (B, gru_hidden)
        return h_next

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution over z given only the deterministic state.

        Used during imagination / dreaming.

        Args:
            h: Deterministic state of shape (B, gru_hidden).

        Returns:
            Tuple of (z, logits) where z is sampled stochastic state (B, stoch_dim)
            and logits has shape (B, stoch_classes, stoch_categoricals).
        """
        logits = self.prior_net(h)                                            # (B, stoch_dim)
        logits = logits.view(-1, self.stoch_classes, self.stoch_categoricals) # (B, C, K)
        z = self._sample_categorical(logits)                                  # (B, stoch_dim)
        return z, logits

    def posterior(
        self, h: torch.Tensor, encoder_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution over z given h and encoder output.

        Used during training with real observations.

        Args:
            h: Deterministic state of shape (B, gru_hidden).
            encoder_out: Encoder output of shape (B, encoder_output_dim).

        Returns:
            Tuple of (z, logits) where z is sampled stochastic state (B, stoch_dim)
            and logits has shape (B, stoch_classes, stoch_categoricals).
        """
        x = torch.cat([h, encoder_out], dim=1)                               # (B, gru_hidden + enc)
        logits = self.posterior_net(x)                                        # (B, stoch_dim)
        logits = logits.view(-1, self.stoch_classes, self.stoch_categoricals) # (B, C, K)
        z = self._sample_categorical(logits)                                  # (B, stoch_dim)
        return z, logits

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distributions with straight-through gradients.

        Uses a manual Gumbel-softmax implementation that clamps uniform samples
        to avoid NaN on MPS (where torch.rand can produce exact 0.0/1.0).

        Args:
            logits: Unnormalized log-probabilities of shape (B, stoch_classes, stoch_categoricals).

        Returns:
            One-hot encoded and flattened samples of shape (B, stoch_dim).
        """
        if self.training:
            # Manual Gumbel-softmax: clamp uniform to avoid log(0) NaN on MPS
            u = torch.rand_like(logits).clamp(1e-8, 1.0 - 1e-8)
            gumbel_noise = -torch.log(-torch.log(u))
            y_soft = F.softmax((logits + gumbel_noise) / 1.0, dim=-1)
            # Straight-through: hard one-hot in forward, soft gradients in backward
            indices = y_soft.argmax(dim=-1)
            onehot = F.one_hot(indices, self.stoch_categoricals).float()
            onehot = onehot + y_soft - y_soft.detach()  # straight-through
        else:
            # Argmax at eval time
            indices = logits.argmax(dim=-1)
            onehot = F.one_hot(indices, self.stoch_categoricals).float()
        # Flatten: (batch, stoch_classes * stoch_categoricals)
        return onehot.view(-1, self.stoch_dim)
