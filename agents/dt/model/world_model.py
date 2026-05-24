"""
Light-weight forward-prediction World Model for MCTS rollouts.

Given (state, action), predicts (next_state, reward, done_prob).
Trained via supervised learning on real game transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """3-layer MLP that approximates the environment's transition dynamics."""

    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        input_dim = state_dim + act_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head_state  = nn.Linear(hidden_dim, state_dim)
        self.head_reward = nn.Linear(hidden_dim, 1)
        self.head_done   = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state:  (B, state_dim)
            action: (B, act_dim)  one-hot
        Returns:
            next_state: (B, state_dim)
            reward:     (B, 1)
            done_prob:  (B, 1)   ∈ [0, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.head_state(x)
        reward     = self.head_reward(x)
        done_prob  = torch.sigmoid(self.head_done(x))
        return next_state, reward, done_prob

    def loss(self, state, action, target_next_state, target_reward, target_done):
        """
        Compute training loss.

        Args:
            state:             (B, state_dim)
            action:            (B, act_dim)
            target_next_state: (B, state_dim)
            target_reward:     (B, 1)
            target_done:       (B, 1)  0 or 1
        Returns:
            total_loss, {'state_loss', 'reward_loss', 'done_loss'}
        """
        pred_state, pred_reward, pred_done = self.forward(state, action)
        state_loss  = F.mse_loss(pred_state, target_next_state)
        reward_loss = F.mse_loss(pred_reward, target_reward)
        done_loss   = F.binary_cross_entropy(pred_done, target_done)
        total = state_loss + reward_loss + done_loss
        return total, {
            'state_loss':  state_loss.item(),
            'reward_loss': reward_loss.item(),
            'done_loss':   done_loss.item(),
        }
