# tests/test_training.py
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.agents.ppo import PPO
from pacman.training.trainer import Trainer, get_device
from pacman.training.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def config():
    c = load_config()
    # Override for fast tests — use minimal architecture
    c["env"]["num_envs"] = 4
    c["env"]["frame_stack"] = 1
    c["rnd"] = {"enabled": False}
    c["network"]["cnn_channels"] = [32, 64, 64]
    c["network"]["head_hidden"] = 128
    c["ppo"]["rollout_steps"] = 16
    c["ppo"]["minibatch_size"] = 16
    c["ppo"]["num_epochs"] = 2
    c["training"]["total_updates"] = 5
    c["training"]["eval_every"] = 3
    c["training"]["eval_episodes"] = 2
    c["training"]["checkpoint_every"] = 2
    c["game"]["max_steps"] = 200
    return c


class TestTrainer:
    def test_short_training_completes(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            # Should complete without errors
            assert (Path(tmpdir) / "checkpoints" / "latest.pt").exists()

    def test_checkpoint_save_load(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=3)

            # Load checkpoint into fresh network (same architecture)
            network2 = ActorCritic()
            meta = load_checkpoint(Path(tmpdir) / "checkpoints", network2)
            assert meta["update"] >= 0

    def test_curriculum_advances(self, config):
        config["curriculum"]["phase_thresholds"] = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            assert trainer.curriculum_phase >= 1

    def test_training_with_rnd(self, config):
        """Verify training works with RND enabled."""
        config["rnd"] = {"enabled": True, "intrinsic_coef": 0.1}
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=3)
            assert trainer.rnd is not None
            assert (Path(tmpdir) / "checkpoints" / "latest.pt").exists()

    def test_training_with_frame_stack(self, config):
        """Verify training works with frame stacking."""
        config["env"]["frame_stack"] = 2
        # Network needs matching grid_channels
        grid_ch = config["env"]["observation_channels"] * 2
        net_cfg = config["network"]
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=3)
            assert (Path(tmpdir) / "checkpoints" / "latest.pt").exists()


class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            network = ActorCritic()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            save_checkpoint(
                Path(tmpdir), 100, network, optimizer,
                {"mean": 1.0, "var": 2.0, "count": 100},
                curriculum_phase=1, config={"test": True},
            )
            network2 = ActorCritic()
            optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-3)
            meta = load_checkpoint(Path(tmpdir), network2, optimizer2)
            assert meta["update"] == 100
            assert meta["curriculum_phase"] == 1

    def test_checkpoint_with_rnd_state(self):
        """Verify RND state is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            network = ActorCritic()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            rnd_state = {
                "model_state_dict": {"test_key": torch.tensor([1.0])},
                "optimizer_state_dict": {},
                "rnd_normalizer": {"mean": 0.5, "var": 1.5, "count": 50},
            }
            save_checkpoint(
                Path(tmpdir), 50, network, optimizer,
                {"mean": 0.0, "var": 1.0, "count": 100},
                curriculum_phase=0, config={},
                rnd_state=rnd_state,
            )
            network2 = ActorCritic()
            meta = load_checkpoint(Path(tmpdir), network2)
            assert meta["rnd_state"] is not None
            assert meta["rnd_state"]["rnd_normalizer"]["mean"] == 0.5
