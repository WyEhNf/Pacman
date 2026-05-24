# pacman/agents/rnd.py
"""Random Network Distillation for intrinsic motivation / curiosity-driven exploration."""
import torch
import torch.nn as nn


class RNDModule(nn.Module):
    """RND provides intrinsic rewards based on prediction error of a fixed random network.

    The target network has random fixed weights. The predictor network is trained
    to match the target's outputs. In novel states, the predictor's error is high,
    providing an intrinsic exploration bonus that drives systematic exploration.
    """

    def __init__(
        self,
        input_channels: int = 8,
        grid_h: int = 31,
        grid_w: int = 28,
        hidden_size: int = 256,
        output_size: int = 128,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        def _make_cnn():
            return nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Flatten(),
            )

        # Compute CNN output size dynamically
        with torch.no_grad():
            flat_size = _make_cnn()(torch.zeros(1, input_channels, grid_h, grid_w)).shape[1]

        # Target: fixed random network (never trained)
        self.target = nn.Sequential(
            _make_cnn(),
            nn.Linear(flat_size, output_size),
        )

        # Predictor: trained to match target (deeper — learns slowly on familiar states)
        self.predictor = nn.Sequential(
            _make_cnn(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Freeze target — its weights never change
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs_grid: torch.Tensor) -> torch.Tensor:
        """Compute per-sample intrinsic reward as MSE(target, predictor).

        Args:
            obs_grid: (N, C, H, W) latest frame observations.

        Returns:
            (N,) intrinsic rewards (unnormalized).
        """
        target_out = self.target(obs_grid)
        pred_out = self.predictor(obs_grid)
        return ((target_out - pred_out) ** 2).mean(dim=1)

    def update(self, obs_grid: torch.Tensor) -> float:
        """Train predictor to match target on a batch of observations.

        Returns scalar loss value.
        """
        with torch.no_grad():
            target_out = self.target(obs_grid)
        pred_out = self.predictor(obs_grid)
        loss = ((target_out - pred_out) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
