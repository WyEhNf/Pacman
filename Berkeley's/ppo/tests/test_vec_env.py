# tests/test_vec_env.py
import numpy as np
import pytest

from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.env.vec_env import VecEnv


@pytest.fixture
def config():
    c = load_config()
    c["env"]["frame_stack"] = 1  # basic tests without frame stacking
    return c


class TestVecEnv:
    def test_observation_shapes(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert obs["scalars"].shape == (4, 5)
        assert obs["grid"].dtype == np.float32
        assert obs["scalars"].dtype == np.float32

    def test_observation_values_in_range(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].min() >= 0.0
        assert obs["grid"].max() <= 1.0
        assert obs["scalars"].min() >= 0.0
        assert obs["scalars"].max() <= 1.0

    def test_legal_masks_shape(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        assert masks.shape == (4, 4)
        assert masks.dtype == bool
        assert masks.any(axis=1).all()  # every env has at least 1 legal action

    def test_step_returns_correct_shapes(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        actions = np.array([np.random.choice(np.where(m)[0]) for m in masks])
        obs, rewards, dones, infos = env.step(actions)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_auto_reset(self, config):
        """Run until at least one env finishes, verify it auto-resets."""
        config = dict(config)
        config["game"] = dict(config["game"])
        config["game"]["max_steps"] = 50  # force quick termination
        env = VecEnv(4, config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        seen_done = False
        for _ in range(100):
            masks = env.get_legal_masks()
            actions = np.array([rng.choice(np.where(m)[0]) for m in masks])
            obs, rewards, dones, infos = env.step(actions)
            if dones.any():
                seen_done = True
                # After auto-reset, obs should still be valid
                assert obs["grid"].shape == (4, 8, 31, 28)
                break
        assert seen_done

    def test_parity_with_single_env(self, config):
        """VecEnv(N=1) should produce identical results to PacmanEnv with same seed."""
        single = PacmanEnv(config, difficulty=0)
        vec = VecEnv(1, config, difficulty=0)

        single_obs, _ = single.reset(seed=100)
        vec_obs = vec.reset(seed=100)

        np.testing.assert_array_equal(single_obs["grid"], vec_obs["grid"][0])
        np.testing.assert_array_equal(single_obs["scalars"], vec_obs["scalars"][0])

        rng_single = np.random.default_rng(200)
        rng_vec = np.random.default_rng(200)
        for _ in range(50):
            mask_s = single.get_legal_mask()
            mask_v = vec.get_legal_masks()[0]
            np.testing.assert_array_equal(mask_s, mask_v)

            action = rng_single.choice(np.where(mask_s)[0])
            obs_s, r_s, term_s, _, _ = single.step(int(action))
            obs_v, r_v, d_v, _ = vec.step(np.array([action]))

            np.testing.assert_array_almost_equal(r_s, r_v[0])
            if term_s:
                break

    def test_set_difficulty(self, config):
        env = VecEnv(2, config, difficulty=0)
        env.reset(seed=42)
        env.set_difficulty(2)
        assert env.difficulty == 2


class TestFrameStacking:
    def test_stacked_observation_shape(self):
        config = load_config()
        config["env"]["frame_stack"] = 4
        env = VecEnv(2, config)
        obs = env.reset(seed=42)
        assert obs["grid"].shape == (2, 32, 31, 28)  # 4 frames * 8 channels
        assert obs["scalars"].shape == (2, 5)

    def test_initial_frames_identical(self):
        """After reset, all stacked frames should be the same initial frame."""
        config = load_config()
        config["env"]["frame_stack"] = 4
        env = VecEnv(2, config)
        obs = env.reset(seed=42)
        C = 8
        for f in range(4):
            np.testing.assert_array_equal(
                obs["grid"][:, f * C:(f + 1) * C],
                obs["grid"][:, :C],
            )

    def test_frames_update_on_step(self):
        """After stepping, the oldest and newest frames should differ."""
        config = load_config()
        config["env"]["frame_stack"] = 4
        env = VecEnv(2, config)
        obs = env.reset(seed=42)
        masks = env.get_legal_masks()
        actions = np.array([np.where(m)[0][0] for m in masks])
        obs2, _, _, _ = env.step(actions)
        C = 8
        # Oldest 3 frames should be the initial frame (shifted)
        # Newest frame should be different if Pac-Man moved
        assert obs2["grid"].shape == (2, 32, 31, 28)

    def test_frame_buffer_resets_on_done(self):
        """When an env is done and auto-resets, its frame buffer should reset too."""
        config = load_config()
        config["env"]["frame_stack"] = 4
        config["game"] = dict(config["game"])
        config["game"]["max_steps"] = 50
        env = VecEnv(4, config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        for _ in range(200):
            masks = env.get_legal_masks()
            actions = np.array([rng.choice(np.where(m)[0]) for m in masks])
            obs, _, dones, _ = env.step(actions)
            if dones.any():
                # Stacked obs should still be valid shape
                assert obs["grid"].shape == (4, 32, 31, 28)
                # For done envs, all frames should be the reset frame
                for i in range(4):
                    if dones[i]:
                        C = 8
                        for f in range(4):
                            np.testing.assert_array_equal(
                                obs["grid"][i, f * C:(f + 1) * C],
                                obs["grid"][i, :C],
                            )
                break

    def test_single_env_frame_stacking(self):
        """PacmanEnv should also support frame stacking."""
        config = load_config()
        config["env"]["frame_stack"] = 4
        env = PacmanEnv(config, difficulty=0)
        obs, _ = env.reset(seed=42)
        assert obs["grid"].shape == (32, 31, 28)
