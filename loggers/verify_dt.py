# verify_dt.py
import torch
from src.model.decision_transformer import DecisionTransformer

dt = DecisionTransformer(state_dim=256, act_dim=5, d_model=128, n_heads=2, n_layers=3, context_len=20)
B, K = 4, 20
rtg    = torch.randn(B, K, 1)
states = torch.randn(B, K, 256)
acts   = torch.randn(B, K, 5)   # one-hot
tsteps = torch.arange(K).unsqueeze(0).expand(B, -1)

action_logits, values, return_preds = dt(rtg, states, acts, tsteps)
print(f"Action logits: {action_logits.shape}")   # (4, 20, 5)
print(f"Values:        {values}")                 # None
print(f"Return preds:  {return_preds.shape}")     # (4, 20, 1)

# Test gradient flow
loss = action_logits.sum()
loss.backward()
print(f"Gradient OK: {dt.embed_rtg.weight.grad is not None}")

# Test inference
single_action = dt.get_action(rtg[:, :1], states[:, :1], acts[:, :1], tsteps[:, :1])
print(f"Inference action logits: {single_action.shape}")  # (5,)