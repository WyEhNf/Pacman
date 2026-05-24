"""Tests for the online dream loop components."""
import torch
import numpy as np
import pytest
from pathlib import Path
from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.dream_trainer import DreamTrainer, DreamPolicy
from pacman.training.wm_trainer import WMTrainer


@pytest.fixture
def config():
    cfg = load_config()
    cfg["env"]["frame_stack"] = 1
    return cfg


@pytest.fixture
def small_wm():
    """A tiny world model for fast tests."""
    wm = WorldModel()
    return wm


class TestDreamTrainerReturnsResult:
    def test_train_returns_dict_with_best_score(self, small_wm, config, tmp_path):
        """DreamTrainer.train() should return a dict with best_score, best_update, best_path."""
        trainer = DreamTrainer(
            world_model=small_wm,
            config=config,
            device=torch.device("cpu"),
            imagination_horizon=3,
            num_imaginations=4,
            lr=1e-3,
            ppo_epochs=1,
        )
        result = trainer.train(
            total_updates=10,
            log_every=5,
            eval_every=5,
            eval_episodes=2,
            save_dir=tmp_path / "dream",
            patience=100,
        )
        assert isinstance(result, dict)
        assert "best_score" in result
        assert "best_update" in result
        assert "best_path" in result
        assert isinstance(result["best_score"], float)
        assert isinstance(result["best_update"], int)
        assert isinstance(result["best_path"], Path)
        assert result["best_path"].exists()


def _make_small_buffer(config, num_episodes=5, steps_per_ep=50):
    """Collect a few short episodes with random actions."""
    env = PacmanEnv(config, difficulty=0)
    buf = EpisodeReplayBuffer(max_episodes=100)
    for i in range(num_episodes):
        env.reset(seed=i)
        grids, scalars, actions, rewards, dones = [], [], [], [], []
        for step in range(steps_per_ep):
            raw = env._build_obs()
            grids.append(raw["grid"])
            scalars.append(raw["scalars"])
            mask = env.get_legal_mask()
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal))
            _, reward, terminated, truncated, _ = env.step(action)
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


class TestDreamDataCollection:
    def test_collect_dream_episodes(self, small_wm, config):
        """Collect episodes using a DreamPolicy + WorldModel in the real env."""
        device = torch.device("cpu")
        wm = small_wm
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)

        policy = DreamPolicy(latent_dim=wm.rssm.latent_dim)
        policy.eval()

        env = PacmanEnv(config, difficulty=0)

        # Collect one episode
        env.reset(seed=0)
        obs = env._build_obs()

        h, z = wm.rssm.initial_state(1)
        h = wm.rssm.dynamics(h, z, torch.zeros(1, dtype=torch.long))

        grids, scalars, actions, rewards, dones = [], [], [], [], []
        done = False
        max_steps = 50

        for step in range(max_steps):
            grid_t = torch.as_tensor(obs["grid"][None])
            scalars_t = torch.as_tensor(obs["scalars"][None])
            enc = wm.encoder(grid_t, scalars_t)
            z, _ = wm.rssm.posterior(h, enc)

            latent = torch.cat([h, z], dim=-1)
            logits, _ = policy(latent)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

            grids.append(obs["grid"])
            scalars.append(obs["scalars"])
            actions.append(action)

            action_t = torch.tensor([action], dtype=torch.long)
            h = wm.rssm.dynamics(h, z, action_t)

            _, reward, terminated, truncated, _ = env.step(action)
            obs = env._build_obs()
            rewards.append(reward)
            dones.append(terminated or truncated)

            if terminated or truncated:
                break

        assert len(grids) > 0
        assert len(grids) == len(actions) == len(rewards) == len(dones)
        grid_arr = np.stack(grids)
        assert grid_arr.shape[1:] == (8, 31, 28)


class TestWMTrainerFineTune:
    def test_fine_tune_loads_checkpoint_and_trains(self, config, tmp_path):
        """fine_tune() should load existing WM checkpoint and train with lower LR."""
        device = torch.device("cpu")
        buf = _make_small_buffer(config)

        # 1. Train initial model for 2 steps to get a checkpoint
        wm = WorldModel().to(device)
        trainer = WMTrainer(wm, buf, device, lr=3e-4, seq_len=10, batch_size=2)
        save_dir = tmp_path / "wm"
        trainer.train(total_steps=2, log_every=1, save_every=2, save_dir=save_dir)

        checkpoint_path = save_dir / "world_model_latest.pt"
        assert checkpoint_path.exists()

        # 2. Fine-tune from checkpoint
        wm2, trainer2 = WMTrainer.fine_tune(
            checkpoint_path=checkpoint_path,
            replay_buffer=buf,
            device=device,
            lr=1e-4,
            seq_len=10,
            batch_size=2,
        )
        assert isinstance(wm2, WorldModel)
        assert isinstance(trainer2, WMTrainer)

        # 3. Run a few training steps — should not error
        trainer2.train(total_steps=2, log_every=1, save_every=2, save_dir=save_dir)


class TestOnlineLoopIntegration:
    def test_one_iteration_runs_without_error(self, config, tmp_path):
        """Run 1 iteration of the online loop with tiny params — smoke test."""
        device = torch.device("cpu")

        # 1. Create initial world model and save it
        wm = WorldModel().to(device)
        wm_dir = tmp_path / "world_model"
        wm_dir.mkdir()
        torch.save(
            {"model_state_dict": wm.state_dict(), "step": 0},
            wm_dir / "world_model_latest.pt",
        )

        # 2. Create a tiny replay buffer and save it
        buf = _make_small_buffer(config, num_episodes=5, steps_per_ep=30)
        buf_path = tmp_path / "replay_buffer.pt"
        buf.save(str(buf_path))

        # 3. Import and run one iteration
        from scripts.train_online import run_online_loop

        result = run_online_loop(
            wm_checkpoint=wm_dir / "world_model_latest.pt",
            buffer_path=buf_path,
            config=config,
            device=device,
            max_iterations=1,
            dream_updates=10,
            dream_eval_every=5,
            dream_eval_episodes=2,
            dream_patience=100,
            dream_horizon=3,
            dream_imaginations=4,
            dream_lr=1e-3,
            collect_episodes=3,
            wm_fine_tune_steps=5,
            wm_lr=1e-4,
            save_dir=tmp_path / "online_run",
        )

        assert isinstance(result, dict)
        assert "iterations_completed" in result
        assert "best_score" in result
        assert result["iterations_completed"] == 1
