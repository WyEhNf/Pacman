"""World Model training loop."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer


class WMTrainer:
    """Trains a WorldModel on data from an EpisodeReplayBuffer.

    Args:
        world_model: The WorldModel to train.
        replay_buffer: Buffer containing collected episode data.
        device: Torch device to train on.
        lr: Learning rate for Adam optimizer.
        grad_clip: Maximum gradient norm for clipping.
        seq_len: Length of sub-sequences sampled from episodes.
        batch_size: Number of sequences per training batch.
    """

    def __init__(
        self,
        world_model: WorldModel,
        replay_buffer: EpisodeReplayBuffer,
        device: torch.device,
        lr: float = 3e-4,
        grad_clip: float = 100.0,
        seq_len: int = 50,
        batch_size: int = 16,
    ) -> None:
        self.wm = world_model
        self.buffer = replay_buffer
        self.device = device
        self.grad_clip = grad_clip
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)

    def train(
        self,
        total_steps: int = 100_000,
        log_every: int = 100,
        save_every: int = 5000,
        save_dir: Path | None = None,
    ) -> None:
        """Run the main training loop.

        Args:
            total_steps: Total number of gradient steps.
            log_every: Print loss summary every N steps.
            save_every: Save checkpoint every N steps.
            save_dir: Directory to save checkpoints. If None, saving is skipped.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        self.wm.train()
        start_time = time.time()
        log_losses: dict[str, float] = {}
        log_count = 0

        for step in range(1, total_steps + 1):
            # 1. Sample batch from buffer
            batch = self.buffer.sample_sequences(self.batch_size, self.seq_len)

            # 2. Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 3. Forward pass
            losses = self.wm.train_step(batch)

            # 4. Backward pass
            self.optimizer.zero_grad()
            losses["_total_tensor"].backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), self.grad_clip)
            self.optimizer.step()

            # Accumulate for logging
            for key in ("total", "recon", "reward", "continue", "kl"):
                log_losses[key] = log_losses.get(key, 0.0) + losses[key]
            log_count += 1

            # 5. Log
            if step % log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                avg = {k: v / log_count for k, v in log_losses.items()}
                print(
                    f"Step {step}/{total_steps} | "
                    f"total={avg['total']:.4f} recon={avg['recon']:.4f} "
                    f"reward={avg['reward']:.4f} continue={avg['continue']:.4f} "
                    f"kl={avg['kl']:.4f} | "
                    f"{steps_per_sec:.1f} steps/s",
                    flush=True,
                )
                log_losses = {}
                log_count = 0

            # 6. Save
            if save_dir is not None and step % save_every == 0:
                self._save(save_dir, step)

        # Final save
        if save_dir is not None:
            self._save(save_dir, total_steps)
            print(f"Final checkpoint saved at step {total_steps}.")

    def _save(self, save_dir: Path, step: int) -> None:
        """Save model and optimizer state."""
        state = {
            "model_state_dict": self.wm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(state, save_dir / "world_model_latest.pt")
        torch.save(state, save_dir / f"world_model_{step}.pt")
        print(f"  Saved checkpoint at step {step}.")

    @classmethod
    def fine_tune(
        cls,
        checkpoint_path: str | Path,
        replay_buffer: EpisodeReplayBuffer,
        device: torch.device,
        lr: float = 1e-4,
        grad_clip: float = 100.0,
        seq_len: int = 50,
        batch_size: int = 16,
    ) -> tuple[WorldModel, "WMTrainer"]:
        """Create a WMTrainer from an existing checkpoint for fine-tuning.

        Loads both model and optimizer state for momentum continuity.

        Args:
            checkpoint_path: Path to a world_model checkpoint (.pt file).
            replay_buffer: Replay buffer (may contain new + old data).
            device: Torch device.
            lr: Learning rate (should be lower than initial training).
            grad_clip: Max gradient norm.
            seq_len: Sequence length for sampling.
            batch_size: Batch size.

        Returns:
            Tuple of (WorldModel, WMTrainer) ready for training.
        """
        checkpoint_path = Path(checkpoint_path)
        wm = WorldModel().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        wm.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(
            world_model=wm,
            replay_buffer=replay_buffer,
            device=device,
            lr=lr,
            grad_clip=grad_clip,
            seq_len=seq_len,
            batch_size=batch_size,
        )

        # Restore optimizer state for momentum continuity
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Override LR to the fine-tune rate
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = lr

        print(f"  Loaded WM checkpoint from {checkpoint_path} (step {checkpoint.get('step', '?')})")
        return wm, trainer
