"""Graph Attention Network — topology encoder for maze graphs."""
import torch, torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.n_heads, self.concat = n_heads, concat
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(1, n_heads, 2 * out_dim))
        self.leaky_relu, self.dropout = nn.LeakyReLU(0.2), nn.Dropout(dropout)
    def forward(self, x, edge_index): raise NotImplementedError

class GATEncoder(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=64, out_dim=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for i in range(n_layers):
            last = (i == n_layers - 1)
            h = out_dim if last else hidden_dim
            self.layers.append(GATLayer(d, h, n_heads, dropout, concat=not last))
            d = h if last else h * n_heads
    def forward(self, x, edge_index): raise NotImplementedError
