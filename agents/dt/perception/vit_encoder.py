"""Vision Transformer — spatial encoder for Pacman maze grids."""
import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=21, patch_size=3, in_chans=3, d_model=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, d_model, patch_size, patch_size)
    def forward(self, x): raise NotImplementedError

class ViTEncoder(nn.Module):
    def __init__(self, img_size=21, patch_size=3, in_chans=3, d_model=256, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): raise NotImplementedError
