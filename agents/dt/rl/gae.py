"""Generalized Advantage Estimation (Schulman et al., 2016).

δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
A_t = δ_t + γλ · (1 - done_t) · A_{t+1}          (reverse recurrence)
G_t = A_t + V(s_t)                                 (target for value function)
"""

import numpy as np


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages and returns from a completed trajectory.

    Args:
        rewards:  (T,)   float array — rewards at each timestep
        values:   (T+1,) float array — V(s₀) ... V(s_{T-1}), V(s_T)
                  V(s_T) should be 0 for terminal states, or the
                  bootstrap value for truncated trajectories.
        dones:    (T,)   float array — 1.0 if episode ended at this
                  step (next state is terminal), 0.0 otherwise.
        gamma:    discount factor
        lam:      GAE λ  (0 = 1-step TD,  1 = Monte Carlo)

    Returns:
        advantages: (T,)   A_t
        returns:    (T,)   G_t = A_t + V(s_t)   (target for critic)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = (rewards[t]
                 + gamma * values[t + 1] * next_non_terminal
                 - values[t])
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns

