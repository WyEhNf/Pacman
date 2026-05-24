"""Fused MLP kernel — GELU + Dropout + Residual + LayerNorm in one pass (placeholder)."""
import torch, torch.nn as nn

class FusedMLP(nn.Module):
    """MLP block that fuses the linear→GELU→dropout→residual→layernorm chain."""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): raise NotImplementedError
