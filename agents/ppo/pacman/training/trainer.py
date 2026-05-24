# pacman/training/trainer.py
"""PPO training orchestrator with frame stacking and RND curiosity."""
import time
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


class _CSVWriter:
    """Fallback metrics logger when TensorBoard is unavailable."""
    def __init__(self, log_dir: str):
        self._path = Path(log_dir) / "metrics.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: list[str] = []
        self._file = None

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        if self._file is None:
            self._file = open(self._path, "w")
            self._file.write("step,tag,value\n")
        self._file.write(f"{global_step},{tag},{value}\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


from ..engine.constants import MAZE_ROWS, MAZE_COLS
from ..env.vec_env import VecEnv
from ..agents.networks import ActorCritic
from ..agents.ppo import PPO
from ..agents.rollout import RolloutBuffer
from ..training.evaluator import Evaluator
from ..training.checkpoint import save_checkpoint, load_checkpoint


class RunningMeanStd:
    """Tracks running mean/std for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        total = self.count + count
        new_mean = self.mean + delta * count / total
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + delta ** 2 * self.count * count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self._clip, self._clip,
        )

    @property
    def _clip(self):
        return 10.0

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


def get_device(config: dict) -> torch.device:
    choice = config.get("device", "auto")
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(choice)


class Trainer:
    def __init__(self, config: dict, run_dir: Path, resume: bool = False):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(config)

        # Frame stacking config
        env_cfg = config["env"]
        self.frame_stack = env_cfg.get("frame_stack", 1)
        self.obs_channels = env_cfg["observation_channels"]
        grid_channels = self.obs_channels * self.frame_stack

        # Environment — difficulty synced with curriculum below
        self.vec_env = VecEnv(env_cfg["num_envs"], config, difficulty=0)
        self._difficulty_synced = False

        # Network — wider architecture with frame-stacked input
        net_cfg = config["network"]
        self.network = ActorCritic(
            grid_channels=grid_channels,
            num_scalars=env_cfg["num_scalar_features"],
            cnn_channels=net_cfg["cnn_channels"],
            cnn_kernels=net_cfg["cnn_kernels"],
            cnn_strides=net_cfg["cnn_strides"],
            shared_hidden=net_cfg["shared_hidden"],
            head_hidden=net_cfg["head_hidden"],
        )

        # PPO
        self.ppo = PPO(self.network, config, self.device)

        # Rollout buffer — uses stacked grid channels
        self.rollout = RolloutBuffer(
            num_envs=env_cfg["num_envs"],
            rollout_steps=config["ppo"]["rollout_steps"],
            grid_shape=(grid_channels, MAZE_ROWS, MAZE_COLS),
            num_scalars=env_cfg["num_scalar_features"],
            num_actions=4,
        )

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # RND intrinsic curiosity
        rnd_cfg = config.get("rnd", {})
        self.use_rnd = rnd_cfg.get("enabled", False)
        self.rnd = None
        self.rnd_normalizer = None
        self.rnd_coef = 0.0
        if self.use_rnd:
            from ..agents.rnd import RNDModule
            self.rnd = RNDModule(
                input_channels=self.obs_channels,
                grid_h=MAZE_ROWS,
                grid_w=MAZE_COLS,
                hidden_size=rnd_cfg.get("hidden_size", 256),
                output_size=rnd_cfg.get("output_size", 128),
                learning_rate=rnd_cfg.get("learning_rate", 1e-4),
            ).to(self.device)
            self.rnd_coef = rnd_cfg.get("intrinsic_coef", 0.1)
            self.rnd_normalizer = RunningMeanStd()

        # Evaluator
        self.evaluator = Evaluator(config)

        # Logging
        tb_dir = str(self.run_dir / "tensorboard")
        if _HAS_TB:
            self.writer = SummaryWriter(tb_dir)
        else:
            print("[WARN] TensorBoard unavailable — logging metrics to CSV")
            self.writer = _CSVWriter(tb_dir)

        # Curriculum
        self.curriculum_phase = 0
        self.best_clear_rate = 0.0
        self.start_update = 0
        self._entropy_boost_until = -1  # update number until which to boost entropy

        if resume:
            ckpt_dir = self.run_dir / "checkpoints"
            if (ckpt_dir / "latest.pt").exists():
                meta = load_checkpoint(ckpt_dir, self.network, self.ppo.optimizer)
                self.start_update = meta["update"] + 1
                self.curriculum_phase = meta.get("curriculum_phase", 0)
                norm_state = meta.get("reward_normalizer", {})
                if norm_state:
                    self.reward_normalizer.load_state_dict(norm_state)
                # Restore RND state
                rnd_state = meta.get("rnd_state")
                if rnd_state and self.rnd is not None:
                    self.rnd.load_state_dict(rnd_state["model_state_dict"])
                    self.rnd.optimizer.load_state_dict(rnd_state["optimizer_state_dict"])
                    if "rnd_normalizer" in rnd_state:
                        self.rnd_normalizer.load_state_dict(rnd_state["rnd_normalizer"])

    def train(self, total_updates: int | None = None) -> None:
        total = total_updates or self.config["training"]["total_updates"]
        train_cfg = self.config["training"]
        ppo_cfg = self.config["ppo"]
        curriculum_cfg = self.config["curriculum"]
        N = self.config["env"]["num_envs"]
        T = ppo_cfg["rollout_steps"]

        obs = self.vec_env.reset(seed=42)
        self.current_update = self.start_update

        for update in range(self.start_update, total):
            self.current_update = update
            t_start = time.time()

            # --- Anneal schedules ---
            self.ppo.anneal_lr(update, total)
            entropy_coef = self._get_entropy_coef(update, total)
            self.ppo.set_entropy_coef(entropy_coef)

            # --- Advance curriculum ---
            self._advance_curriculum(update, curriculum_cfg)

            # --- Collect rollout ---
            self.rollout.reset()
            for _step in range(T):
                legal_masks = self.vec_env.get_legal_masks()
                actions, log_probs, values = self.ppo.select_action(
                    obs["grid"], obs["scalars"], legal_masks,
                )
                next_obs, rewards, dones, infos = self.vec_env.step(actions)

                # Add intrinsic curiosity rewards from RND
                if self.use_rnd:
                    latest_frame = next_obs["grid"][:, -self.obs_channels:]
                    grid_t = torch.as_tensor(
                        latest_frame.copy(), device=self.device,
                    )
                    raw_intrinsic = self.rnd.compute_intrinsic_reward(grid_t)
                    raw_intrinsic = raw_intrinsic.cpu().numpy()
                    # Normalize by running std (not mean-centered)
                    self.rnd_normalizer.update(raw_intrinsic)
                    norm_intrinsic = raw_intrinsic / (np.sqrt(self.rnd_normalizer.var) + 1e-8)
                    norm_intrinsic = np.clip(norm_intrinsic, -5.0, 5.0)
                    rewards = rewards + self.rnd_coef * norm_intrinsic

                # Normalize combined rewards
                self.reward_normalizer.update(rewards)
                norm_rewards = self.reward_normalizer.normalize(rewards)

                self.rollout.insert(
                    obs["grid"], obs["scalars"],
                    actions, log_probs, values,
                    norm_rewards, dones, legal_masks,
                )
                obs = next_obs

            # --- Compute GAE ---
            legal_masks = self.vec_env.get_legal_masks()
            last_values = self.ppo.get_value(obs["grid"], obs["scalars"], legal_masks)
            self.rollout.compute_gae(
                last_values, ppo_cfg["gamma"], ppo_cfg["gae_lambda"],
            )

            # --- PPO Update ---
            metrics = self.ppo.update(self.rollout)

            # --- RND Update ---
            if self.use_rnd:
                rnd_loss = self._update_rnd()
                metrics["rnd_loss"] = rnd_loss

            # --- Logging ---
            fps = N * T / (time.time() - t_start)
            if update % train_cfg["log_every"] == 0:
                self.writer.add_scalar("loss/policy", metrics["policy_loss"], update)
                self.writer.add_scalar("loss/value", metrics["value_loss"], update)
                self.writer.add_scalar("loss/entropy", metrics["entropy"], update)
                self.writer.add_scalar("schedule/learning_rate",
                                       self.ppo.optimizer.param_groups[0]["lr"], update)
                self.writer.add_scalar("schedule/entropy_coef", entropy_coef, update)
                self.writer.add_scalar("schedule/curriculum_phase",
                                       self.curriculum_phase, update)
                self.writer.add_scalar("throughput/fps", fps, update)
                if self.use_rnd:
                    self.writer.add_scalar("rnd/loss", metrics.get("rnd_loss", 0), update)
                    self.writer.add_scalar("rnd/reward_var", self.rnd_normalizer.var, update)

            # Progress to stdout every 10 updates (or at eval points)
            if update % 10 == 0:
                print(f"[Update {update:>5d}/{total}] "
                      f"p_loss={metrics['policy_loss']:.4f} "
                      f"v_loss={metrics['value_loss']:.4f} "
                      f"ent={metrics['entropy']:.4f} "
                      f"lr={self.ppo.optimizer.param_groups[0]['lr']:.2e} "
                      f"fps={fps:.0f}")

            # --- Evaluation ---
            is_best = False
            if update % train_cfg["eval_every"] == 0 and update > 0:
                eval_result = self.evaluator.evaluate(
                    self.network, train_cfg["eval_episodes"], self.device,
                )
                self.writer.add_scalar("perf/clear_3life",
                                       eval_result["clear_rate_3life"], update)
                self.writer.add_scalar("perf/score_3life",
                                       eval_result["mean_score_3life"], update)
                self.writer.add_scalar("perf/clear_1life",
                                       eval_result["clear_rate_1life"], update)
                self.writer.add_scalar("perf/score_1life",
                                       eval_result["mean_score_1life"], update)
                self.writer.add_scalar("perf/ghosts_3life",
                                       eval_result["mean_ghosts_3life"], update)
                self.writer.add_scalar("perf/ghosts_1life",
                                       eval_result["mean_ghosts_1life"], update)
                is_best = eval_result["clear_rate_1life"] > self.best_clear_rate
                if is_best:
                    self.best_clear_rate = eval_result["clear_rate_1life"]
                print(f"[Update {update:>5d}] "
                      f"3life: clear={eval_result['clear_rate_3life']:.1%} "
                      f"score={eval_result['mean_score_3life']:.0f} "
                      f"ghosts={eval_result['mean_ghosts_3life']:.1f} | "
                      f"1life: clear={eval_result['clear_rate_1life']:.1%} "
                      f"score={eval_result['mean_score_1life']:.0f} "
                      f"ghosts={eval_result['mean_ghosts_1life']:.1f} "
                      f"fps={fps:.0f}")

            # --- Checkpoint ---
            if update % train_cfg["checkpoint_every"] == 0 and update > 0:
                self._save(update, is_best=is_best)

        # Final checkpoint
        self._save(total - 1, is_best=is_best)
        self.writer.close()

    def _save(self, update: int, is_best: bool = False) -> None:
        """Save checkpoint including RND state if active."""
        rnd_state = None
        if self.use_rnd:
            rnd_state = {
                "model_state_dict": self.rnd.state_dict(),
                "optimizer_state_dict": self.rnd.optimizer.state_dict(),
                "rnd_normalizer": self.rnd_normalizer.state_dict(),
            }
        save_checkpoint(
            self.run_dir / "checkpoints", update,
            self.network, self.ppo.optimizer,
            self.reward_normalizer.state_dict(),
            self.curriculum_phase, self.config,
            is_best=is_best, rnd_state=rnd_state,
        )

    def _update_rnd(self) -> float:
        """Update RND predictor on observations from the rollout."""
        rnd_cfg = self.config.get("rnd", {})
        proportion = rnd_cfg.get("update_proportion", 0.25)

        # Extract latest frames from stored stacked observations
        all_grids = self.rollout.obs_grids  # (T, N, stacked_C, H, W)
        latest = all_grids[:, :, -self.obs_channels:]  # (T, N, 8, H, W)
        T, N = latest.shape[:2]
        total = T * N
        flat = latest.reshape(total, self.obs_channels, MAZE_ROWS, MAZE_COLS)

        # Sample subset for efficiency
        num_samples = max(int(total * proportion), 64)
        indices = np.random.choice(total, size=min(num_samples, total), replace=False)
        batch = torch.as_tensor(flat[indices].copy(), device=self.device)

        return self.rnd.update(batch)

    def _get_entropy_coef(self, update: int, total: int) -> float:
        cfg = self.config["ppo"]
        start = cfg["entropy_coef_start"]
        end = cfg["entropy_coef_end"]
        anneal_frac = cfg["entropy_anneal_fraction"]
        if update < total * anneal_frac:
            coef = start
        else:
            progress = (update - total * anneal_frac) / (total * (1 - anneal_frac))
            coef = start + (end - start) * min(progress, 1.0)

        # Entropy boost after curriculum transitions — force re-exploration
        if update < self._entropy_boost_until:
            coef = max(coef, start * 2)

        return coef

    def _advance_curriculum(self, update: int, curriculum_cfg: dict) -> None:
        thresholds = curriculum_cfg["phase_thresholds"]
        difficulties = curriculum_cfg["difficulties"]
        # Find the highest phase whose threshold has been reached
        target_phase = 0
        for phase, threshold in enumerate(thresholds):
            if update >= threshold:
                target_phase = phase
        # Sync difficulty on first call (handles resume where phase loaded from ckpt)
        if not self._difficulty_synced:
            self._difficulty_synced = True
            self.vec_env.set_difficulty(difficulties[target_phase])
        if target_phase != self.curriculum_phase:
            self.curriculum_phase = target_phase
            self.vec_env.set_difficulty(difficulties[target_phase])
            # Boost entropy for 300 updates after phase transition
            self._entropy_boost_until = update + 300
            print(f"[Update {update}] Curriculum -> phase {target_phase} "
                  f"(difficulty={difficulties[target_phase]}) "
                  f"[entropy boost until {update + 300}]")
