# pacman/agents/networks.py
"""CNN Actor-Critic network for Pac-Man PPO."""
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared CNN backbone with separate policy and value heads."""

    def __init__(
        self,
        grid_channels: int = 8,
        grid_h: int = 31,
        grid_w: int = 28,
        num_scalars: int = 5,
        num_actions: int = 4,
        cnn_channels: list[int] = (32, 64, 64),
        cnn_kernels: list[int] = (3, 3, 3),
        cnn_strides: list[int] = (1, 2, 2),
        shared_hidden: int = 512,
        head_hidden: int = 128,
    ):
        super().__init__()
        # CNN encoder
        layers = []
        in_ch = grid_channels
        for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_h, grid_w)
            cnn_out = self.cnn(dummy).view(1, -1).shape[1]

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(cnn_out + num_scalars, shared_hidden),
            nn.ReLU(),
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Policy head final layer: small init for uniform exploration
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        # Value head final layer: standard init
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(
        self,
        grid: torch.Tensor,      # (N, 8, 31, 28)
        scalars: torch.Tensor,    # (N, 4)
        legal_mask: torch.Tensor, # (N, 4) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        cnn_out = self.cnn(grid).flatten(1)
        combined = torch.cat([cnn_out, scalars], dim=1)
        shared = self.shared(combined)
        logits = self.policy(shared)
        # Mask illegal actions
        logits = logits.masked_fill(~legal_mask, -1e8)
        value = self.value(shared)
        return logits, value
