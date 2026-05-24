"""End-to-end test: collect data -> train world model -> imagine."""
import torch
import numpy as np
import pytest
from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.world_model.world_model import WorldModel


@pytest.fixture
def config():
    cfg = load_config()
    # Use frame_stack=1 so _build_obs returns raw 8-channel grids
    cfg["env"]["frame_stack"] = 1
    return cfg


@pytest.fixture
def small_buffer(config):
    """Collect 5 short episodes with random actions."""
    env = PacmanEnv(config, difficulty=0)
    buf = EpisodeReplayBuffer(max_episodes=100)
    for i in range(5):
        obs, _ = env.reset(seed=i)
        grids, scalars, actions, rewards, dones = [], [], [], [], []
        for step in range(50):
            raw = env._build_obs()
            grids.append(raw["grid"])
            scalars.append(raw["scalars"])
            mask = env.get_legal_mask()
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break
        buf.add_episode({
            "grid": torch.as_tensor(np.stack(grids)),
            "scalars": torch.as_tensor(np.stack(scalars)),
            "action": torch.tensor(actions, dtype=torch.long),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
        })
    return buf


class TestDreamingE2E:
    def test_world_model_trains_without_error(self, small_buffer):
        """Create WorldModel, train for 3 steps on small_buffer, verify losses are positive and finite."""
        wm = WorldModel()
        wm.train()
        optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)

        for _ in range(3):
            batch = small_buffer.sample_sequences(batch_size=2, seq_len=10)
            losses = wm.train_step(batch)

            # Verify all expected loss keys exist
            for key in ("recon", "reward", "continue", "kl", "total"):
                assert key in losses, f"Missing loss key: {key}"
                val = losses[key]
                assert np.isfinite(val), f"Loss '{key}' is not finite: {val}"
                assert val >= 0, f"Loss '{key}' is negative: {val}"

            # Backward pass should succeed
            total_tensor = losses["_total_tensor"]
            assert isinstance(total_tensor, torch.Tensor)
            assert total_tensor.requires_grad
            optimizer.zero_grad()
            total_tensor.backward()
            optimizer.step()

    def test_imagination_produces_valid_output(self, small_buffer, config):
        """Create WorldModel in eval mode, imagine from a real observation, verify shapes and no NaNs."""
        wm = WorldModel()
        # Train for 1 step so weights are not entirely random zeros
        wm.train()
        batch = small_buffer.sample_sequences(batch_size=2, seq_len=10)
        losses = wm.train_step(batch)
        losses["_total_tensor"].backward()

        wm.eval()

        # Get a real starting observation from PacmanEnv
        env = PacmanEnv(config, difficulty=0)
        env.reset(seed=42)
        raw = env._build_obs()
        start_grid = torch.as_tensor(raw["grid"]).unsqueeze(0)      # (1, 8, 31, 28)
        start_scalars = torch.as_tensor(raw["scalars"]).unsqueeze(0)  # (1, 5)

        horizon = 5

        def random_action_fn(h, z):
            return torch.randint(0, 4, (h.shape[0],))

        result = wm.imagine(start_grid, start_scalars, random_action_fn, horizon=horizon)

        # Verify shapes
        assert result["h"].shape == (1, horizon, 512), (
            f"Expected h shape (1,{horizon},512), got {result['h'].shape}"
        )
        assert result["z"].shape == (1, horizon, 2048), (
            f"Expected z shape (1,{horizon},2048), got {result['z'].shape}"
        )
        assert result["reward"].shape == (1, horizon), (
            f"Expected reward shape (1,{horizon}), got {result['reward'].shape}"
        )
        assert result["cont"].shape == (1, horizon), (
            f"Expected cont shape (1,{horizon}), got {result['cont'].shape}"
        )

        # Verify no NaNs
        for key in ("h", "z", "reward", "cont"):
            assert not torch.isnan(result[key]).any(), f"NaN found in result['{key}']"

    def test_decode_produces_valid_grid(self, small_buffer):
        """Create WorldModel in eval mode, decode random latents, verify grid shape and no NaNs."""
        wm = WorldModel()
        wm.eval()

        h = torch.randn(1, 512)
        z = torch.randn(1, 2048)
        grid, scalars = wm.decode(h, z)

        assert grid.shape == (1, 8, 31, 28), (
            f"Expected grid shape (1,8,31,28), got {grid.shape}"
        )
        assert scalars.shape == (1, 5), (
            f"Expected scalars shape (1,5), got {scalars.shape}"
        )
        assert not torch.isnan(grid).any(), "NaN found in decoded grid"
        assert not torch.isnan(scalars).any(), "NaN found in decoded scalars"
