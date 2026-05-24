"""PPO adapter — wraps DecisionTransformer for online fine-tuning."""
import torch
import torch.nn.functional as F


class PPOAdapter:
    def __init__(self, dt_model, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.dt = dt_model
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.dt.configure_ppo()  # attaches predict_value head

    # -----------------------------------------------------------------
    #  Rollout: sample one action at the current timestep
    # -----------------------------------------------------------------

    def act(self, rtg, states, actions, timesteps):
        """
        Sample an action at the LAST timestep of the context window.

        Args:
            rtg:       (1, K, 1)
            states:    (1, K, state_dim)
            actions:   (1, K, act_dim)
            timesteps: (1, K)

        Returns:
            action:     int          chosen action id
            log_prob:   float        log π(action | context)
            value:      float        V(state) at the last timestep
        """
        action_logits, values, _ = self.dt(rtg, states, actions, timesteps)
        # Take the last timestep
        logits = action_logits[0, -1, :]           # (act_dim,)
        value  = values[0, -1, 0].item()           # scalar

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()

        return action.item(), log_prob, value

    # -----------------------------------------------------------------
    #  Training: compute PPO loss on a batch of experiences
    # -----------------------------------------------------------------

    def evaluate(self, rtg, states, actions, timesteps,
                 old_log_probs, advantages, returns):
        """
        Compute PPO clipped loss on a batch of context windows.

        Args:
            rtg:           (B, K, 1)
            states:        (B, K, state_dim)
            actions:       (B, K, act_dim)  one-hot
            timesteps:     (B, K)
            old_log_probs: (B, K)           log π_old at each state token
            advantages:    (B, K)           GAE advantages
            returns:       (B, K)           target values  G_t

        Returns:
            total_loss: scalar tensor
            stats:      dict with {policy_loss, value_loss, entropy, approx_kl}
        """
        B, K = actions.shape[0], actions.shape[1]

        # Mask out padding (timestep 0 entries are padding at the front)
        mask = (timesteps > 0).float()   # (B, K)

        action_logits, values, _ = self.dt(rtg, states, actions, timesteps)
        # action_logits: (B, K, act_dim)
        # values:        (B, K, 1)

        # --- Policy loss ---
        logits_flat = action_logits.reshape(B * K, -1)
        actions_flat = actions.reshape(B * K, -1).argmax(dim=-1)  # one-hot → id

        probs = F.softmax(logits_flat, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_flat).reshape(B, K)  # (B, K)

        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)            # (B, K)

        # Clipped surrogate
        adv = advantages
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / mask.sum().clamp(min=1)

        # --- Value loss ---
        value_loss = F.mse_loss(values.squeeze(-1), returns, reduction='none')
        value_loss = (value_loss * mask).sum() / mask.sum().clamp(min=1)

        # --- Entropy bonus ---
        entropy = dist.entropy().reshape(B, K)
        entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)

        # --- Total loss ---
        total_loss = (policy_loss
                      + self.value_coef * value_loss
                      - self.entropy_coef * entropy)

        # --- KL approx for monitoring ---
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean()

        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss':  value_loss.item(),
            'entropy':     entropy.item(),
            'approx_kl':   approx_kl.item(),
        }
        return total_loss, stats
