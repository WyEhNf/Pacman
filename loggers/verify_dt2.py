import torch
from src.model.decision_transformer import DecisionTransformer

dt = DecisionTransformer(state_dim=128, act_dim=5, d_model=64, n_heads=2, n_layers=2, context_len=10)
B, K = 4, 10
rtg    = torch.randn(B, K, 1)
states = torch.randn(B, K, 128)
acts   = torch.randn(B, K, 5)
tsteps = torch.arange(K).unsqueeze(0).expand(B, -1)

# Test 1: shapes
action_logits, values, return_preds = dt(rtg, states, acts, tsteps)
assert action_logits.shape == (B, K, 5), f"FAIL shape: {action_logits.shape}"
assert values is None, f"FAIL values: {values}"
print(f"[PASS] Action logits: {action_logits.shape}")

# Test 2: gradient flow
loss = action_logits.sum()
loss.backward()
assert dt.embed_rtg.weight.grad is not None, "FAIL rtg grad"
assert dt.embed_state.weight.grad is not None, "FAIL state grad"
print("[PASS] Gradient flows to all embeddings")

# Test 3: inference
single = dt.get_action(rtg[:, :1], states[:, :1], acts[:, :1], tsteps[:, :1])
assert single.shape == (5,), f"FAIL inference shape: {single.shape}"
print(f"[PASS] Inference: {single.shape}")

# Test 4: causal mask
dt.eval()
y1 = dt(rtg[:, :2], states[:, :2], acts[:, :2], tsteps[:, :2])[0]
rtg_mod = rtg[:, :2].clone(); rtg_mod[0, 1, 0] = 999.0
y2 = dt(rtg_mod, states[:, :2], acts[:, :2], tsteps[:, :2])[0]
diff = (y1[0, 0, :] - y2[0, 0, :]).abs().max().item()
assert diff < 1e-5, f"FAIL causal: diff={diff:.8f}"
print(f"[PASS] Causal mask: diff={diff:.1e}")

# Test 5: configure_ppo
dt.configure_ppo()
assert dt.predict_value is not None, "FAIL ppo config"
_, values, _ = dt(rtg[:, :1], states[:, :1], acts[:, :1], tsteps[:, :1])
assert values.shape == (B, 1, 1), f"FAIL value shape: {values.shape}"
print("[PASS] PPO value head")

# Test 6: memory check
n = sum(p.numel() for p in dt.parameters())
print(f"\nAll tests passed. Model: {n:,} params, ~{n*4/1024/1024:.1f} MB")
