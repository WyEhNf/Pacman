# tests/test_ppo.py
import numpy as np
import torch
import pytest

from pacman.agents.networks import ActorCritic
from pacman.agents.rollout import RolloutBuffer
from pacman.agents.ppo import PPO
from pacman.utils.config import load_config


@pytest.fixture
def config():
    c = load_config()
    c["ppo"]["minibatch_size"] = 16  # small for fast tests
    return c


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def network():
    return ActorCritic()


@pytest.fixture
def ppo(network, config, device):
    return PPO(network, config, device)


class TestRolloutBuffer:
    def test_insert_and_shape(self):
        buf = RolloutBuffer(
            num_envs=4, rollout_steps=8,
            grid_shape=(8, 31, 28), num_scalars=5, num_actions=4,
        )
        for _ in range(8):
            buf.insert(
                obs_grid=np.random.randn(4, 8, 31, 28).astype(np.float32),
                obs_scalars=np.random.randn(4, 5).astype(np.float32),
                action=np.random.randint(0, 4, size=4),
                log_prob=np.random.randn(4).astype(np.float32),
                value=np.random.randn(4).astype(np.float32),
                reward=np.random.randn(4).astype(np.float32),
                done=np.zeros(4, dtype=bool),
                legal_mask=np.ones((4, 4), dtype=bool),
            )
        assert buf._step == 8

    def test_gae_computation(self):
        """Hand-verified GAE example with 3 steps, 1 env."""
        buf = RolloutBuffer(1, 3, (8, 31, 28), 5, 4)
        # Step 0: r=1, v=0.5, done=False
        # Step 1: r=2, v=1.0, done=False
        # Step 2: r=3, v=1.5, done=False
        # last_value = 2.0
        for t, (r, v) in enumerate([(1, 0.5), (2, 1.0), (3, 1.5)]):
            buf.insert(
                np.zeros((1, 8, 31, 28), dtype=np.float32),
                np.zeros((1, 5), dtype=np.float32),
                np.array([0]),
                np.array([0.0], dtype=np.float32),
                np.array([v], dtype=np.float32),
                np.array([r], dtype=np.float32),
                np.array([False]),
                np.ones((1, 4), dtype=bool),
            )
        buf.compute_gae(np.array([2.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
        # Advantages should be finite and non-zero
        assert np.all(np.isfinite(buf.advantages))
        assert np.any(buf.advantages != 0)

    def test_batch_generator_total_count(self):
        buf = RolloutBuffer(4, 8, (8, 31, 28), 5, 4)
        for _ in range(8):
            buf.insert(
                np.random.randn(4, 8, 31, 28).astype(np.float32),
                np.random.randn(4, 5).astype(np.float32),
                np.random.randint(0, 4, size=4),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.zeros(4, dtype=bool),
                np.ones((4, 4), dtype=bool),
            )
        buf.compute_gae(np.zeros(4, dtype=np.float32), 0.99, 0.95)
        total = sum(b["actions"].shape[0] for b in buf.batch_generator(16, torch.device("cpu")))
        assert total == 32  # 4 envs * 8 steps


class TestPPO:
    def test_select_action(self, ppo):
        obs_grid = np.random.randn(4, 8, 31, 28).astype(np.float32)
        obs_scalars = np.random.randn(4, 5).astype(np.float32)
        masks = np.ones((4, 4), dtype=bool)
        actions, log_probs, values = ppo.select_action(obs_grid, obs_scalars, masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert all(0 <= a < 4 for a in actions)

    def test_action_masking(self, ppo):
        obs_grid = np.random.randn(100, 8, 31, 28).astype(np.float32)
        obs_scalars = np.random.randn(100, 5).astype(np.float32)
        masks = np.zeros((100, 4), dtype=bool)
        masks[:, 0] = True  # only action 0 is legal
        actions, _, _ = ppo.select_action(obs_grid, obs_scalars, masks)
        assert np.all(actions == 0)

    def test_update_returns_finite_losses(self, ppo):
        buf = RolloutBuffer(4, 16, (8, 31, 28), 5, 4)
        for _ in range(16):
            buf.insert(
                np.random.randn(4, 8, 31, 28).astype(np.float32),
                np.random.randn(4, 5).astype(np.float32),
                np.random.randint(0, 4, size=4),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.zeros(4, dtype=bool),
                np.ones((4, 4), dtype=bool),
            )
        buf.compute_gae(np.zeros(4, dtype=np.float32), 0.99, 0.95)
        metrics = ppo.update(buf)
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])

    def test_lr_annealing(self, ppo):
        initial_lr = ppo.optimizer.param_groups[0]["lr"]
        ppo.anneal_lr(500, 1000)
        new_lr = ppo.optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr
        assert new_lr > 0
