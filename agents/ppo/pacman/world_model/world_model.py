# pacman/world_model/world_model.py
"""Integrated WorldModel: encoder, RSSM, decoder, and prediction heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pacman.world_model.encoder import ObservationEncoder
from pacman.world_model.decoder import ObservationDecoder
from pacman.world_model.heads import RewardHead, ContinueHead
from pacman.world_model.rssm import RSSM


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential (inverse of symlog): sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class WorldModel(nn.Module):
    """Integrated world model combining encoder, RSSM dynamics, decoder, and heads.

    Provides train_step() for supervised training on observation sequences and
    imagine() for latent-space rollouts used by a dream agent.

    Args:
        grid_channels: Number of grid observation channels.
        num_scalars: Number of scalar observation features.
        stoch_classes: Number of classes per categorical distribution in RSSM.
        stoch_categoricals: Number of categorical distributions in RSSM.
        gru_hidden: Dimension of the GRU hidden state in RSSM.
        action_dim: Number of discrete actions.
        free_nats: Free nats threshold for KL loss clamping.
        kl_beta: Scaling factor for the KL divergence loss.
    """

    def __init__(
        self,
        grid_channels: int = 8,
        num_scalars: int = 5,
        stoch_classes: int = 32,
        stoch_categoricals: int = 64,
        gru_hidden: int = 512,
        action_dim: int = 4,
        free_nats: float = 1.0,
        kl_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.free_nats = free_nats
        self.kl_beta = kl_beta

        # Sub-modules
        self.encoder = ObservationEncoder(
            grid_channels=grid_channels,
            num_scalars=num_scalars,
            output_dim=512,
        )
        self.rssm = RSSM(
            stoch_classes=stoch_classes,
            stoch_categoricals=stoch_categoricals,
            gru_hidden=gru_hidden,
            action_dim=action_dim,
            encoder_output_dim=512,
        )
        self.decoder = ObservationDecoder(
            latent_dim=self.rssm.latent_dim,
            grid_channels=grid_channels,
            num_scalars=num_scalars,
        )
        self.reward_head = RewardHead(latent_dim=self.rssm.latent_dim)
        self.continue_head = ContinueHead(latent_dim=self.rssm.latent_dim)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float | torch.Tensor]:
        """Run a single training step on a batch of observation sequences.

        Args:
            batch: Dictionary with keys:
                - grid: (B, T, C, H, W) grid observations
                - scalars: (B, T, S) scalar features
                - action: (B, T) discrete actions
                - reward: (B, T) rewards
                - done: (B, T) episode termination flags

        Returns:
            Dictionary with string keys for each loss component (.item() values),
            plus "_total_tensor" holding the differentiable total loss tensor.
        """
        grid = batch["grid"]
        scalars = batch["scalars"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        B, T = grid.shape[0], grid.shape[1]

        # 1. Encode all observations: reshape to (B*T, ...), encode, reshape back
        grid_flat = grid.reshape(B * T, *grid.shape[2:])      # (B*T, C, H, W)
        scalars_flat = scalars.reshape(B * T, *scalars.shape[2:])  # (B*T, S)
        enc_flat = self.encoder(grid_flat, scalars_flat)       # (B*T, 512)
        enc_seq = enc_flat.reshape(B, T, -1)                   # (B, T, 512)

        # 2. Roll out RSSM for T steps
        h, z = self.rssm.initial_state(B)
        h_list, z_list = [], []
        prior_logits_list, post_logits_list = [], []

        for t in range(T):
            h = self.rssm.dynamics(h, z, action[:, t])
            _, prior_logits = self.rssm.prior(h)
            z, post_logits = self.rssm.posterior(h, enc_seq[:, t])

            h_list.append(h)
            z_list.append(z)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)

        h_seq = torch.stack(h_list, dim=1)                   # (B, T, gru_hidden)
        z_seq = torch.stack(z_list, dim=1)                   # (B, T, stoch_dim)

        # 3. Concatenate latent = cat([h, z], dim=-1) -> (B, T, latent_dim)
        latent_seq = torch.cat([h_seq, z_seq], dim=-1)       # (B, T, 2560)

        # 4. Decode: flatten to (B*T, latent_dim)
        latent_flat = latent_seq.reshape(B * T, -1)
        grid_pred, scalars_pred = self.decoder(latent_flat)   # (B*T, C, H, W), (B*T, S)
        reward_pred = self.reward_head(latent_flat)           # (B*T, 1)
        continue_pred = self.continue_head(latent_flat)       # (B*T, 1)

        # Reshape predictions back
        grid_pred = grid_pred.reshape(B, T, *grid_pred.shape[1:])
        scalars_pred = scalars_pred.reshape(B, T, -1)
        reward_pred = reward_pred.reshape(B, T)
        continue_pred = continue_pred.reshape(B, T)

        # 5. Compute losses
        # Reconstruction loss
        recon_loss = (
            F.mse_loss(grid_pred, grid)
            + F.mse_loss(scalars_pred, scalars)
        )

        # Reward loss (predict symlog of reward)
        reward_loss = F.mse_loss(reward_pred, symlog(reward))

        # Continue loss
        continue_target = (~done).float()
        continue_loss = F.binary_cross_entropy_with_logits(continue_pred, continue_target)

        # KL divergence between posterior and prior (categorical distributions)
        prior_logits_seq = torch.stack(prior_logits_list, dim=1)  # (B, T, C, K)
        post_logits_seq = torch.stack(post_logits_list, dim=1)    # (B, T, C, K)

        eps = 1e-8
        post_probs = F.softmax(post_logits_seq, dim=-1).clamp(min=eps)
        prior_probs = F.softmax(prior_logits_seq, dim=-1).clamp(min=eps)

        # KL(posterior || prior) per categorical, summed over classes, averaged over batch/time
        kl_per_cat = (post_probs * (post_probs.log() - prior_probs.log())).sum(dim=-1)  # (B, T, C)
        kl = kl_per_cat.sum(dim=-1).mean()  # sum over categoricals, mean over (B, T)

        # Free nats clamp
        kl_clamped = torch.clamp(kl - self.free_nats, min=0.0)

        # Total loss
        total = recon_loss + reward_loss + continue_loss + self.kl_beta * kl_clamped

        return {
            "recon": recon_loss.item(),
            "reward": reward_loss.item(),
            "continue": continue_loss.item(),
            "kl": kl.item(),
            "total": total.item(),
            "_total_tensor": total,
        }

    @torch.no_grad()
    def imagine(
        self,
        start_grid: torch.Tensor,
        start_scalars: torch.Tensor,
        action_fn,
        horizon: int = 15,
    ) -> dict[str, torch.Tensor]:
        """Roll out the model in latent space from a starting observation.

        Args:
            start_grid: Starting grid observation of shape (B, C, H, W).
            start_scalars: Starting scalar features of shape (B, S).
            action_fn: Callable(h, z) -> action tensor of shape (B,).
            horizon: Number of imagination steps.

        Returns:
            Dictionary with keys:
            - h: (B, horizon, gru_hidden) deterministic states
            - z: (B, horizon, stoch_dim) stochastic states
            - reward: (B, horizon) predicted rewards
            - cont: (B, horizon) predicted continuation probabilities
        """
        B = start_grid.shape[0]

        # 1. Encode starting observation
        enc = self.encoder(start_grid, start_scalars)  # (B, 512)

        # 2. Initialize h, z and get posterior z from encoder output
        h, z = self.rssm.initial_state(B)
        h = self.rssm.dynamics(h, z, torch.zeros(B, dtype=torch.long, device=h.device))
        z, _ = self.rssm.posterior(h, enc)

        # 3. Imagination rollout
        h_list, z_list, reward_list, cont_list = [], [], [], []

        for _ in range(horizon):
            action = action_fn(h, z)
            h = self.rssm.dynamics(h, z, action)
            z, _ = self.rssm.prior(h)  # Use prior during imagination

            latent = torch.cat([h, z], dim=-1)  # (B, latent_dim)
            reward = self.reward_head(latent).squeeze(-1)      # (B,)
            cont = self.continue_head(latent).squeeze(-1)      # (B,)
            cont = torch.sigmoid(cont)                          # probabilities

            h_list.append(h)
            z_list.append(z)
            reward_list.append(reward)
            cont_list.append(cont)

        return {
            "h": torch.stack(h_list, dim=1),          # (B, horizon, gru_hidden)
            "z": torch.stack(z_list, dim=1),          # (B, horizon, stoch_dim)
            "reward": torch.stack(reward_list, dim=1), # (B, horizon)
            "cont": torch.stack(cont_list, dim=1),    # (B, horizon)
        }

    def decode(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent state into grid and scalar observations.

        Args:
            h: Deterministic state of shape (B, gru_hidden).
            z: Stochastic state of shape (B, stoch_dim).

        Returns:
            Tuple of (grid, scalars) where grid has shape (B, C, H, W)
            and scalars has shape (B, S).
        """
        latent = torch.cat([h, z], dim=-1)
        return self.decoder(latent)
