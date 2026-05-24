"""
GPT-2 style Transformer Block.

Pre-LN architecture:
    x = x + MHA(LayerNorm(x))    ← masked / unmasked self-attention
    x = x + FFN(LayerNorm(x))    ← GELU feed-forward with 4x expansion

Used as the backbone for DecisionTransformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Masked / unmasked multi-head self-attention (scaled dot-product)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V projection in a single matrix for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """
        Args:
            x:           (B, T, d_model)
            causal_mask: if True, each position only attends to ≤ itself
        Returns:
            (B, T, d_model)
        """
        B, T, C = x.shape

        # --- 1. Linear projection + split heads ---
        qkv = self.qkv(x)                     # (B, T, 3·d_model)
        q, k, v = qkv.chunk(3, dim=-1)        # each (B, T, d_model)

        # reshape: (B, T, d_model) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # --- 2. Scaled dot-product attention ---
        scale = math.sqrt(self.d_k)
        attn_scores = (q @ k.transpose(-2, -1)) / scale   # (B, n_heads, T, T)

        if causal_mask:
            # Upper triangle mask: position i can only see positions ≤ i
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # --- 3. Weighted sum ---
        attn_out = attn_weights @ v                    # (B, n_heads, T, d_k)

        # --- 4. Merge heads + final projection ---
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, T, n_heads, d_k)
        attn_out = attn_out.view(B, T, C)                  # (B, T, d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU, 4x expansion in the hidden layer."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """One GPT-2-style decoder block (Pre-LN)."""

    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        # Pre-LN: norm BEFORE the sublayer, then residual
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.ln2(x))
        return x
