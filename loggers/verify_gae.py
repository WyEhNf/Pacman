import numpy as np
from src.rl.gae import compute_gae

# Test 1: simple 3-step trajectory, no termination
rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
values  = np.array([0.0, 0.5, 0.8, 0.0], dtype=np.float32)
dones   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=0.95)
print(f"Test 1 advantages: {adv}")
print(f"Test 1 returns:    {ret}")

# Test 2: terminal state at end
rewards2 = np.array([1.0, 10.0], dtype=np.float32)
values2  = np.array([0.0, 0.5, 0.0], dtype=np.float32)
dones2   = np.array([0.0, 1.0], dtype=np.float32)
adv2, ret2 = compute_gae(rewards2, values2, dones2, gamma=0.99, lam=1.0)
print(f"Test 2 advantages: {adv2}")
print(f"Test 2 returns:    {ret2}")

# Test 3: verify returns = advantages + values[:T]
assert np.allclose(ret, adv + values[:3]), "FAIL: returns != A + V"
assert np.allclose(ret2, adv2 + values2[:2]), "FAIL test2: returns != A + V"
print("All tests passed.")
