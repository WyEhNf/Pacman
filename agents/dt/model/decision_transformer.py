"""
Decision Transformer (Chen et al., NeurIPS 2021).

Models (Return-to-Go, state, action) triplets as a causal sequence.
At each state-token position, predicts the next action.

Input layout: [R₀, S₀, A₀,  R₁, S₁, A₁,  ...,  Rₖ, Sₖ]
               │         │                │
               └─ 3 tokens per timestep ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock


class DecisionTransformer(nn.Module):
    """
    Phase 1: Behavioral Cloning — CE loss on predicted actions
    Phase 2: PPO fine-tuning     — adds a Value head
    """

    def __init__(self, state_dim: int, act_dim: int,
                 d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, d_ff: int = None,
                 context_len: int = 20, max_timestep: int = 4096,
                 dropout: float = 0.1):
        super().__init__()
        self.state_dim, self.act_dim = state_dim, act_dim
        self.d_model = d_model
        self.context_len = context_len

        d_ff = d_ff or 4 * d_model

        # --- Embedding layers ---
        self.embed_rtg   = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)     # action as one-hot
        self.embed_timestep = nn.Embedding(max_timestep, d_model)

        # --- Pre-LN + Transformer backbone ---
        self.ln_pre = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # --- Prediction heads ---
        self.predict_action = nn.Linear(d_model, act_dim)
        self.predict_return = nn.Linear(d_model, 1)
        self.predict_value  = None   # attached later for PPO

        self._init_weights()

    def _init_weights(self):
        for pn, p in self.named_parameters():
            if p.dim() >= 2 and 'embed' not in pn:
                nn.init.normal_(p, mean=0.0, std=0.02)

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------

    def forward(self, rtg, states, actions, timesteps,
                attention_mask=None):
        """
        Args:
            rtg:       (B, K, 1)         return-to-go
            states:    (B, K, state_dim)
            actions:   (B, K, act_dim)   one-hot
            timesteps: (B, K)             step indices
            attention_mask: (B, 3·K)     1=real, 0=pad  (optional)

        Returns:
            action_logits:  (B, K, act_dim)
            values:         (B, K, 1) or None
            return_preds:   (B, K, 1) or None
        """
        B, K, _ = rtg.shape

        # --- 1. Embed each modality ---
        rtg_emb   = self.embed_rtg(rtg)                 # (B, K, d_model)
        state_emb = self.embed_state(states)             # (B, K, d_model)

        # Handle both one-hot (float) and integer action inputs
        if actions.dim() == 2 or actions.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # (B, K) integer → one-hot
            actions_oh = torch.zeros(B, K, self.act_dim, device=actions.device)
            actions_oh.scatter_(-1, actions.long().unsqueeze(-1), 1.0)
        else:
            actions_oh = actions.float()

        act_emb   = self.embed_action(actions_oh)        # (B, K, d_model)
        time_emb  = self.embed_timestep(timesteps)       # (B, K, d_model)

        # Add timestep embedding to every modality
        rtg_emb   = rtg_emb   + time_emb
        state_emb = state_emb + time_emb
        act_emb   = act_emb   + time_emb

        # --- 2. Interleave into [R₀, S₀, A₀, R₁, S₁, A₁, ...] ---
        # stack → (B, 3, K, d_model) → permute → (B, K, 3, d_model) → flatten
        tokens = torch.stack([rtg_emb, state_emb, act_emb], dim=1)
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, 3 * K, self.d_model)
        tokens = self.ln_pre(tokens)

        # Build attention mask if provided
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]   # (B, 1, 1, 3K)
            attn_mask = (1.0 - attn_mask) * -1e9            # 0 → -inf
        else:
            attn_mask = None

        # --- 3. Pass through transformer blocks ---
        for block in self.blocks:
            tokens = block(tokens, causal_mask=True)

        # --- 4. Reshape back to (B, 3, K, d_model) ---
        tokens = tokens.reshape(B, K, 3, self.d_model).permute(0, 2, 1, 3)
        #   idx 0 = rtg tokens,   idx 1 = state tokens,   idx 2 = action tokens

        action_logits = self.predict_action(tokens[:, 1, :, :])   # (B, K, act_dim)
        return_preds  = self.predict_return(tokens[:, 2, :, :])   # (B, K, 1)

        values = None
        if self.predict_value is not None:
            values = self.predict_value(tokens[:, 1, :, :])       # (B, K, 1)

        return action_logits, values, return_preds

    # -----------------------------------------------------------------
    #  Inference helpers
    # -----------------------------------------------------------------

    def get_action(self, rtg, states, actions, timesteps):
        """Inference: return action logits at the LAST state token.

        Args:
            rtg:       (1, K, 1)
            states:    (1, K, state_dim)
            actions:   (1, K, act_dim)
            timesteps: (1, K)
        Returns:
            (act_dim,)  logits
        """
        with torch.no_grad():
            action_logits, _, _ = self.forward(
                rtg, states, actions, timesteps)
        return action_logits[0, -1, :]

    def configure_ppo(self):
        """Attach a Value head for PPO fine-tuning."""
        self.predict_value = nn.Linear(self.d_model, 1)
