"""Flash Attention 2 — IO-aware exact attention (placeholder)."""
import torch

def flash_attention(Q, K, V, causal=True):
    """Drop-in replacement for softmax(QK^T/√d)V using tiled SRAM algorithm.
    Fall back to PyTorch's scaled_dot_product_attention when Triton is unavailable."""
    try:
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=causal)
    except Exception:
        raise NotImplementedError("Triton Flash Attention kernel not implemented yet")
