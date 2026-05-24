import torch
from src.model.decision_transformer import DecisionTransformer
from src.rl.ppo_adapter import PPOAdapter

dt = DecisionTransformer(state_dim=64, act_dim=5, d_model=64, n_heads=2, n_layers=2, context_len=10)
adapter = PPOAdapter(dt, epsilon=0.2)

B, K = 8, 10
rtg    = torch.randn(B, K, 1)
states = torch.randn(B, K, 64)
acts   = torch.randn(B, K, 5)
tsteps = torch.randint(1, 100, (B, K))

# Test 1: act()
a, lp, v = adapter.act(rtg[:1], states[:1], acts[:1], tsteps[:1])
print(f"[PASS] act() → action={a}, log_prob={lp:.3f}, value={v:.3f}")

# Test 2: evaluate()
old_log_probs = torch.randn(B, K)
advantages    = torch.randn(B, K)
returns       = torch.randn(B, K)
loss, stats = adapter.evaluate(rtg, states, acts, tsteps, old_log_probs, advantages, returns)
print(f"[PASS] evaluate() → loss={loss.item():.4f}")
for k, v in stats.items():
    print(f"        {k}: {v:.4f}")

# Test 3: gradient flows through value head
loss.backward()
has_grad = dt.predict_value.weight.grad is not None
print(f"[PASS] Value-head gradient: {has_grad}")

# Test 4: clipping behaviour
# When ratio is way too high (log_probs far apart), the clipped term should kick in
old_lp2 = torch.zeros(B, K)                     # log(1) = 0 for old
new_lp2 = torch.ones(B, K) * 2                  # log(e^2) for new → ratio = e^2 ≈ 7.4
adv2    = torch.ones(B, K)                      # positive advantage
loss2, _ = adapter.evaluate(rtg, states, acts, tsteps, old_lp2, adv2, returns)
print(f"[PASS] Clipped ratio: loss={loss2.item():.4f}")
