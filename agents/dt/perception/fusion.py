"""Multi-modal fusion → single state vector for Decision Transformer."""
import torch, torch.nn as nn

class PerceptionFusion(nn.Module):
    def __init__(self, vit_dim=256, gat_dim=128, gh_h=11, gh_w=20, n_ghosts=4, scalar_dim=16, d_model=256):
        super().__init__()
        total = vit_dim + gat_dim + gh_h * gh_w * n_ghosts + scalar_dim
        self.proj = nn.Linear(total, d_model)
    def forward(self, vit_cls, gat_feat, ghost_heatmaps, scalars): raise NotImplementedError
