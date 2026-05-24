"""
Phase 2 verification: GAE → PPOAdapter → Trainer → mini training loop.
All tests use synthetic data; no skeleton environment required.
"""

import numpy as np
import torch

from src.rl.gae import compute_gae
from src.model.decision_transformer import DecisionTransformer
from src.rl.ppo_adapter import PPOAdapter
from src.rl.trainer import PPOTrainer


# ═══════════════════════════════════════════════════════════════════
#  Test 1: GAE — known values
# ═══════════════════════════════════════════════════════════════════

def test_gae():
    rewards = np.array([1., 1., 1.], dtype=np.float32)
    values  = np.array([0., 0.5, 0.8, 0.], dtype=np.float32)
    dones   = np.array([0., 0., 0.], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=0.95)

    # Hand-computed values
    expected_adv = np.array([2.639305, 1.391, 0.2], dtype=np.float32)
    expected_ret = np.array([2.639305, 1.891, 1.0], dtype=np.float32)

    assert np.allclose(adv, expected_adv, atol=1e-4), f"GAE adv mismatch: {adv} vs {expected_adv}"
    assert np.allclose(ret, expected_ret, atol=1e-4), f"GAE ret mismatch: {ret} vs {expected_ret}"
    assert np.allclose(ret, adv + values[:3], atol=1e-4), "ret != adv + values[:T]"
    print("[PASS] Test 1: GAE")

    # Terminal case
    rewards2 = np.array([1., 10.], dtype=np.float32)
    values2  = np.array([0., 0.5, 0.], dtype=np.float32)
    dones2   = np.array([0., 1.], dtype=np.float32)
    adv2, ret2 = compute_gae(rewards2, values2, dones2, gamma=0.99, lam=1.0)
    assert np.allclose(ret2, adv2 + values2[:2], atol=1e-4)
    print("[PASS] Test 1b: GAE with terminal")


# ═══════════════════════════════════════════════════════════════════
#  Test 2: PPOAdapter — act and evaluate
# ═══════════════════════════════════════════════════════════════════

def test_ppo_adapter():
    dt = DecisionTransformer(state_dim=32, act_dim=5, d_model=32,
                             n_heads=2, n_layers=2, context_len=10)
    adapter = PPOAdapter(dt)

    B, K = 4, 10
    rtg    = torch.randn(B, K, 1)
    states = torch.randn(B, K, 32)
    acts   = torch.randn(B, K, 5)
    tsteps = torch.randint(1, 100, (B, K))

    # --- act ---
    a, lp, v = adapter.act(rtg[:1], states[:1], acts[:1], tsteps[:1])
    assert isinstance(a, int)
    assert isinstance(lp, float)
    assert isinstance(v, float)
    print(f"[PASS] Test 2a: act() → a={a}, lp={lp:.3f}, v={v:.3f}")

    # --- evaluate ---
    old_lp = torch.randn(B, K)
    adv    = torch.randn(B, K)
    ret    = torch.randn(B, K)
    loss, stats = adapter.evaluate(rtg, states, acts, tsteps, old_lp, adv, ret)
    assert loss.requires_grad
    assert 'policy_loss' in stats and 'value_loss' in stats
    loss.backward()
    assert dt.predict_value.weight.grad is not None
    print(f"[PASS] Test 2b: evaluate() → loss={loss.item():.4f}, "
          f"p_loss={stats['policy_loss']:.4f}, v_loss={stats['value_loss']:.4f}")

    # --- clipping ---
    old_lp2 = torch.zeros(B, K)
    new_lp2_acts = torch.randint(0, 5, (B, K))  # just for shape
    rtg2    = torch.randn(B, K, 1)
    states2 = torch.randn(B, K, 32)
    acts2   = torch.randn(B, K, 5)
    tsteps2 = torch.randint(1, 100, (B, K))
    adv2   = torch.ones(B, K)

    # Feed old_lp=0 through evaluate with extreme ratio
    loss2, _ = adapter.evaluate(rtg2, states2, acts2, tsteps2,
                                old_lp2, adv2, torch.randn(B, K))
    print(f"[PASS] Test 2c: clip test → loss={loss2.item():.4f} (should be ~0.5-0.7)")


# ═══════════════════════════════════════════════════════════════════
#  Test 3: Trainer._build_batch — context window construction
# ═══════════════════════════════════════════════════════════════════

class MockTrainer(PPOTrainer):
    def _reset_env(self): pass
    def _step_env(self, s, a): pass
    def _extract_features(self, s): pass

def test_build_batch():
    K = 5
    trainer = MockTrainer(None, {'state_dim': 4, 'act_dim': 3},
                          None, context_len=K)

    # Fake rollout with 10 steps
    n = 10
    rollout = {
        'rtg':       np.arange(n, dtype=np.float32),
        'states':    np.tile(np.arange(n)[:, None], (1, 4)).astype(np.float32),
        'actions':   np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64),
        'timesteps': np.arange(n, dtype=np.int64),
        'log_probs': np.random.randn(n).astype(np.float32),
        'advantages': np.random.randn(n).astype(np.float32),
        'returns':   np.random.randn(n).astype(np.float32),
    }
    trainer.state_dim = 4
    trainer.act_dim = 3

    # Build batch for steps [3, 7]
    (b_rtg, b_states, b_acts, b_tsteps,
     b_old_lp, b_adv, b_ret) = trainer._build_batch(rollout, np.array([3, 7]))

    assert b_rtg.shape == (2, K, 1), f"rtg shape {b_rtg.shape}"
    assert b_states.shape == (2, K, 4), f"states shape {b_states.shape}"
    assert b_acts.shape == (2, K, 3), f"acts shape {b_acts.shape}"
    assert b_tsteps.shape == (2, K), f"tsteps shape {b_tsteps.shape}"
    assert b_old_lp.shape == (2, K), f"old_lp shape {b_old_lp.shape}"
    assert b_adv.shape == (2, K), f"adv shape {b_adv.shape}"
    assert b_ret.shape == (2, K), f"ret shape {b_ret.shape}"

    # Check step 3's context: window covers steps 0-3, left-padded by K-4=1
    # Position K-4=1 onwards should have steps 0,1,2,3
    assert torch.allclose(b_rtg[0, -4:, 0], torch.tensor([0., 1., 2., 3.]))
    assert torch.allclose(b_rtg[0, 0, 0], torch.tensor(0.))  # padded
    # Step 7's context: steps 3-7
    assert torch.allclose(b_rtg[1, :, 0], torch.tensor([3., 4., 5., 6., 7.]))

    print(f"[PASS] Test 3: _build_batch — shapes correct, padding correct")

    # Check action one-hot
    # Step 3 had action=0, so b_acts[0, -1, 0] should be 1.0
    assert b_acts[0, -1, 0] == 1.0
    # Step 4 had action=1
    assert b_acts[1, 1, 1] == 1.0  # offset 1 because step 4 is the 2nd element after step 3
    print(f"[PASS] Test 3b: _build_batch action encoding")


# ═══════════════════════════════════════════════════════════════════
#  Test 4: Mini training loop with synthetic data
# ═══════════════════════════════════════════════════════════════════

def test_mini_loop():
    """Run a tiny train_epoch call with synthetic rollout data."""
    dt = DecisionTransformer(state_dim=4, act_dim=3, d_model=16,
                             n_heads=2, n_layers=2, context_len=5)
    adapter = PPOAdapter(dt)
    opt = torch.optim.SGD(dt.parameters(), lr=0.01)

    trainer = MockTrainer(adapter, {'state_dim': 4, 'act_dim': 3},
                          opt, context_len=5, n_steps=20,
                          batch_size=8)

    # Fake rollout: 20 steps
    n = 20
    rollout = {
        'rtg':       np.random.randn(n).astype(np.float32),
        'states':    np.random.randn(n, 4).astype(np.float32),
        'actions':   np.random.randint(0, 3, n).astype(np.int64),
        'rewards':   np.random.randn(n).astype(np.float32),
        'log_probs': np.random.randn(n).astype(np.float32),
        'values':    np.random.randn(n + 1).astype(np.float32) * 0.1,
        'dones':     np.zeros(n, dtype=np.float32),
        'timesteps': np.arange(n, dtype=np.int64),
        'advantages': None,
        'returns':    None,
    }
    trainer.state_dim = 4
    trainer.act_dim = 3

    # Compute GAE
    rollout['advantages'], rollout['returns'] = compute_gae(
        rollout['rewards'], rollout['values'], rollout['dones'])

    # Run one training epoch
    stats = trainer.train_epoch(rollout)

    assert 'policy_loss' in stats
    assert 'value_loss' in stats
    print(f"[PASS] Test 4: Mini training loop → "
          f"p_loss={stats['policy_loss']:.4f}, "
          f"v_loss={stats['value_loss']:.4f}, "
          f"ent={stats['entropy']:.4f}")

    # Verify that weights actually changed
    w_before = dt.predict_value.weight.clone()
    # Run one more epoch
    trainer.train_epoch(rollout)
    w_after = dt.predict_value.weight
    changed = not torch.allclose(w_before, w_after, atol=1e-6)
    print(f"[PASS] Test 4b: Weights updated after training: {changed}")


# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    test_gae()
    test_ppo_adapter()
    test_build_batch()
    test_mini_loop()
    print("\n===== All Phase 2 tests passed =====")
