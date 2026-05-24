"""
Main Pacman Agent — assembles DT, MCTS, and feature extraction.

Decision flow per step:
  1. extract_features(state) → flat vector
  2. Build context window from last K steps of history
  3. DT forward → action_logits, value
  4. If max(softmax(logits)) > confidence_threshold:
       return argmax(logits)          ← fast path
     Else:
       return MCTS.search(state)     ← slow, deliberate path
"""

import numpy as np
import torch

from src.planning.mcts import MCTS


class PacmanAgent:
    """
    Full-stack Pacman AI agent backed by Decision Transformer + MCTS.

    Usage:
        agent = PacmanAgent(dt_model, world_model, config)
        action = agent.act(game_state)         # called each step
        agent.observe(reward, next_state)      # updates internal buffers
    """

    def __init__(self, dt_model, world_model,
                 context_len=20, state_dim=256, act_dim=5,
                 target_rtg=500.0, confidence_threshold=0.8,
                 mcts_simulations=100, device='cpu'):
        self.dt = dt_model
        self.dt.eval()
        self.context_len = context_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.target_rtg = target_rtg
        self.confidence_threshold = confidence_threshold
        self.device = device

        # MCTS planner (created once, reused)
        self.mcts = MCTS(
            world_model, dt_model,
            state_dim=state_dim, act_dim=act_dim,
            n_simulations=mcts_simulations, device=device,
        ) if world_model is not None else None

        self._reset()

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def act(self, game_state, feature_extractor, legal_actions=None):
        """
        Choose an action for the current game state.

        Args:
            game_state:       skeleton GameState object
            feature_extractor: callable(state) → (state_dim,) numpy array
            legal_actions:    list[int]  (default: all actions)

        Returns:
            action_id: int
            debug_info: dict  with 'confidence', 'mcts_triggered', 'rtg', etc.
        """
        feat = feature_extractor(game_state).astype(np.float32)
        self._history_states.append(feat)
        self._history_actions.append(np.zeros(self.act_dim, dtype=np.float32))
        self._history_rtgs.append(self.current_rtg)

        # Build context window
        rtg_ctx, s_ctx, a_ctx, t_ctx = self._build_context()

        with torch.no_grad():
            rtg_t = torch.FloatTensor(rtg_ctx).unsqueeze(0).unsqueeze(-1).to(self.device)
            s_t   = torch.FloatTensor(s_ctx).unsqueeze(0).to(self.device)
            a_t   = torch.FloatTensor(a_ctx).unsqueeze(0).to(self.device)
            t_t   = torch.LongTensor(t_ctx).unsqueeze(0).to(self.device)

            action_logits, values, _ = self.dt(rtg_t, s_t, a_t, t_t)
            logits = action_logits[0, -1, :].cpu().numpy()
            value  = values[0, -1, 0].item() if values is not None else 0.0

        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        confidence = probs.max()

        # Fast path vs MCTS
        if (confidence >= self.confidence_threshold or
                self.mcts is None or legal_actions == []):
            action = int(np.argmax(logits))
            mcts_triggered = False
        else:
            action, _ = self.mcts.search(feat, legal_actions=legal_actions)
            mcts_triggered = True

        # Update history with actual action
        one_hot = np.zeros(self.act_dim, dtype=np.float32)
        one_hot[action] = 1.0
        self._history_actions[-1] = one_hot
        self._last_value = value

        debug = {
            'action': action,
            'confidence': float(confidence),
            'mcts_triggered': mcts_triggered,
            'value': value,
            'rtg': self.current_rtg,
        }
        return action, debug

    def observe(self, reward):
        """Update internal state after executing an action."""
        self.current_rtg -= reward

    def reset(self):
        """Call at the start of each episode."""
        self._reset()

    # -----------------------------------------------------------------
    #  Internal
    # -----------------------------------------------------------------

    def _reset(self):
        self.current_rtg = self.target_rtg
        self._history_states  = []
        self._history_actions = []
        self._history_rtgs    = []
        self._last_value = 0.0

    def _build_context(self):
        """Build (K, ...) context arrays from history."""
        K = self.context_len
        D = self.state_dim
        A = self.act_dim

        ctx_s = np.array(self._history_states[-K:], dtype=np.float32)
        ctx_a = np.array(self._history_actions[-K:], dtype=np.float32)
        ctx_r = np.array(self._history_rtgs[-K:], dtype=np.float32)
        ctx_t = np.arange(len(ctx_s), dtype=np.int64)

        if len(ctx_s) < K:
            pad = K - len(ctx_s)
            ctx_s = np.pad(ctx_s, ((pad, 0), (0, 0)))
            ctx_a = np.pad(ctx_a, ((pad, 0), (0, 0)))
            ctx_r = np.pad(ctx_r, (pad, 0))
            ctx_t = np.pad(ctx_t, (pad, 0))

        return ctx_r, ctx_s, ctx_a, ctx_t
