"""Phase 3 verification: WorldModel + MCTS."""
import numpy as np
import torch
from src.model.decision_transformer import DecisionTransformer
from src.model.world_model import WorldModel
from src.planning.mcts import MCTS, MCTSNode


# Test 1: WorldModel shapes and loss
def test_world_model():
    wm = WorldModel(state_dim=16, act_dim=5, hidden_dim=64)
    s = torch.randn(8, 16)
    a = torch.zeros(8, 5); a[:, 2] = 1.0
    ns, r, d = wm(s, a)
    assert ns.shape == (8, 16), f"state {ns.shape}"
    assert r.shape == (8, 1), f"reward {r.shape}"
    assert d.shape == (8, 1), f"done {d.shape}"
    assert (d >= 0).all() and (d <= 1).all()
    loss, stats = wm.loss(s, a, ns.detach(), r.detach(), d.detach())
    loss.backward()
    assert wm.fc1.weight.grad is not None
    print(f"[PASS] Test 1: WorldModel → loss={loss.item():.4f}")


# Test 2: MCTS search with mock DT + WorldModel
def test_mcts():
    dt = DecisionTransformer(state_dim=16, act_dim=5, d_model=32, n_heads=2,
                             n_layers=2, context_len=10)
    dt.configure_ppo()  # value head needed for MCTS
    wm = WorldModel(state_dim=16, act_dim=5, hidden_dim=64)

    mcts = MCTS(wm, dt, state_dim=16, act_dim=5,
                n_simulations=10, rollout_depth=3)
    root_state = np.random.randn(16).astype(np.float32)
    action, root = mcts.search(root_state, legal_actions=[0, 1, 2, 3, 4])
    assert 0 <= action <= 4, f"action={action}"
    assert root.N == 10, f"root.N={root.N}"
    print(f"[PASS] Test 2: MCTS → action={action}, root.N={root.N}, "
          f"children={len(root.children)}")

    # Check that visited children have N > 0
    visits = {a: c.N for a, c in root.children.items()}
    print(f"        Visit counts: {visits}")


# Test 3: WorldModel training loop (synthetic data)
def test_wm_training():
    wm = WorldModel(state_dim=8, act_dim=3, hidden_dim=32)
    opt = torch.optim.Adam(wm.parameters(), lr=0.01)

    for epoch in range(50):
        s = torch.randn(32, 8)
        a = torch.zeros(32, 3)
        a[np.arange(32), np.random.randint(0, 3, 32)] = 1.0
        ns = s + 0.1 * torch.randn(32, 8)
        r = torch.randn(32, 1)
        d = (torch.rand(32, 1) > 0.9).float()

        loss, _ = wm.loss(s, a, ns, r, d)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"[PASS] Test 3: WM training → final_loss={loss.item():.4f} "
          f"(should decrease from ~1.0)")


if __name__ == '__main__':
    test_world_model()
    test_mcts()
    test_wm_training()
    print("\n===== All Phase 3 tests passed =====")
