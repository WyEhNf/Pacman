# scripts/train_online.py
"""Online Dream Loop: iteratively train dream agent, collect data, fine-tune world model."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.dream_trainer import DreamTrainer, DreamPolicy
from pacman.training.wm_trainer import WMTrainer
from pacman.training.dream_collector import collect_dream_episodes


def run_online_loop(
    wm_checkpoint: str | Path,
    buffer_path: str | Path,
    config: dict,
    device: torch.device,
    max_iterations: int = 5,
    # Dream training params
    dream_updates: int = 3000,
    dream_eval_every: int = 50,
    dream_eval_episodes: int = 20,
    dream_patience: int = 500,
    dream_horizon: int = 10,
    dream_imaginations: int = 512,
    dream_lr: float = 3e-5,
    # Data collection params
    collect_episodes: int = 300,
    # WM fine-tune params
    wm_fine_tune_steps: int = 20_000,
    wm_lr: float = 1e-4,
    # Output
    save_dir: str | Path | None = None,
) -> dict:
    """Run the online dream loop.

    Args:
        wm_checkpoint: Path to initial world model checkpoint.
        buffer_path: Path to initial replay buffer.
        config: Game/env config dict.
        device: Torch device.
        max_iterations: Maximum number of loop iterations.
        dream_updates: Max dream training updates per iteration.
        dream_eval_every: Dream eval frequency.
        dream_eval_episodes: Episodes per dream eval.
        dream_patience: Early stopping patience for dream training.
        dream_horizon: Imagination rollout horizon.
        dream_imaginations: Number of parallel imagined trajectories.
        dream_lr: Dream agent learning rate.
        collect_episodes: Episodes to collect per iteration.
        wm_fine_tune_steps: WM gradient steps per fine-tune.
        wm_lr: WM fine-tuning learning rate.
        save_dir: Directory for all outputs.

    Returns:
        Dict with iterations_completed, best_score, best_iteration, best_dream_agent_path.
    """
    wm_checkpoint = Path(wm_checkpoint)
    buffer_path = Path(buffer_path)
    if save_dir is None:
        save_dir = wm_checkpoint.parent.parent / "online_loop"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load replay buffer — reset capacity to 10K since load() restores the
    # original max_episodes from the saved file (likely 1000 from PPO collection).
    print(f"Loading replay buffer from {buffer_path} ...")
    buffer = EpisodeReplayBuffer(max_episodes=10_000)
    buffer.load(str(buffer_path))
    buffer._max_episodes = 10_000
    print(f"  Episodes: {len(buffer)} | Steps: {buffer.total_steps}")

    current_wm_path = wm_checkpoint
    overall_best_score = -float("inf")
    overall_best_iteration = -1
    overall_best_agent_path = None
    prev_best = -float("inf")
    iterations_completed = 0
    loop_start = time.time()

    try:
        for iteration in range(max_iterations):
            iter_start = time.time()
            iter_dir = save_dir / f"iter_{iteration}"
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration}")
            print(f"{'='*60}")

            # --- Phase 1: Train dream agent ---
            print(f"\n[Phase 1] Training dream agent (max {dream_updates} updates)...")
            wm = WorldModel().to(device)
            ckpt = torch.load(current_wm_path, map_location=device, weights_only=True)
            wm.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded WM from {current_wm_path}")

            dream_save_dir = iter_dir / "dream_agent"
            trainer = DreamTrainer(
                world_model=wm,
                config=config,
                device=device,
                imagination_horizon=dream_horizon,
                num_imaginations=dream_imaginations,
                lr=dream_lr,
            )
            dream_result = trainer.train(
                total_updates=dream_updates,
                eval_every=dream_eval_every,
                eval_episodes=dream_eval_episodes,
                save_dir=dream_save_dir,
                patience=dream_patience,
            )

            iter_best = dream_result["best_score"]
            print(f"\n  Iteration {iteration} best score: {iter_best:.1f}")

            # Track overall best
            if iter_best > overall_best_score:
                overall_best_score = iter_best
                overall_best_iteration = iteration
                overall_best_agent_path = dream_result["best_path"]

            iterations_completed = iteration + 1

            # Convergence check after Phase 1: no point collecting data and
            # fine-tuning the WM with a worse agent — exit early to save hours.
            if iteration > 0 and iter_best <= prev_best:
                print(
                    f"\n  Converged: iteration {iteration} ({iter_best:.1f}) "
                    f"did not beat iteration {iteration - 1} ({prev_best:.1f})"
                )
                break
            prev_best = iter_best

            # --- Phase 2: Collect data with dream agent ---
            print(f"\n[Phase 2] Collecting {collect_episodes} episodes with dream agent...")
            wm.eval()
            for p in wm.parameters():
                p.requires_grad_(False)

            policy = DreamPolicy(latent_dim=wm.rssm.latent_dim).to(device)
            da_ckpt = torch.load(
                dream_result["best_path"], map_location=device, weights_only=True
            )
            policy.load_state_dict(da_ckpt["policy_state_dict"])

            new_buffer = collect_dream_episodes(
                world_model=wm,
                policy=policy,
                config=config,
                device=device,
                num_episodes=collect_episodes,
            )

            # Merge into main buffer
            for ep in new_buffer._episodes:
                buffer.add_episode(ep)
            print(f"  Buffer now: {len(buffer)} episodes, {buffer.total_steps} steps")

            # Save expanded buffer
            expanded_buf_path = save_dir / f"replay_buffer_iter{iteration}.pt"
            buffer.save(str(expanded_buf_path))
            print(f"  Saved buffer to {expanded_buf_path}")

            # --- Phase 3: Fine-tune world model ---
            print(f"\n[Phase 3] Fine-tuning world model ({wm_fine_tune_steps} steps, lr={wm_lr})...")
            wm_ft, ft_trainer = WMTrainer.fine_tune(
                checkpoint_path=current_wm_path,
                replay_buffer=buffer,
                device=device,
                lr=wm_lr,
            )

            wm_save_dir = iter_dir / "world_model"
            ft_trainer.train(
                total_steps=wm_fine_tune_steps,
                log_every=100,
                save_every=5000,
                save_dir=wm_save_dir,
            )

            # Update current WM path for next iteration
            current_wm_path = wm_save_dir / "world_model_latest.pt"
            if not current_wm_path.exists():
                raise RuntimeError(
                    f"Fine-tuned WM checkpoint not found at {current_wm_path}. "
                    f"WMTrainer.train() may not have saved (steps={wm_fine_tune_steps})."
                )

            iter_elapsed = time.time() - iter_start
            print(f"\n  Iteration {iteration} complete in {iter_elapsed / 60:.1f} min")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving current best...")

    total_elapsed = time.time() - loop_start
    print(f"\n{'='*60}")
    print(f"  ONLINE LOOP COMPLETE")
    print(f"  Iterations: {iterations_completed}")
    print(f"  Best score: {overall_best_score:.1f} (iteration {overall_best_iteration})")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    if overall_best_agent_path:
        print(f"  Best agent: {overall_best_agent_path}")
    print(f"{'='*60}")

    return {
        "iterations_completed": iterations_completed,
        "best_score": float(overall_best_score),
        "best_iteration": overall_best_iteration,
        "best_dream_agent_path": overall_best_agent_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Online Dream Loop")
    parser.add_argument("--world-model", type=str, required=True,
                        help="Path to initial world model checkpoint")
    parser.add_argument("--buffer", type=str, required=True,
                        help="Path to initial replay buffer")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--dream-updates", type=int, default=3000)
    parser.add_argument("--collect-episodes", type=int, default=300)
    parser.add_argument("--wm-steps", type=int, default=20_000)
    parser.add_argument("--wm-lr", type=float, default=1e-4)
    parser.add_argument("--dream-lr", type=float, default=3e-5)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    run_online_loop(
        wm_checkpoint=args.world_model,
        buffer_path=args.buffer,
        config=config,
        device=device,
        max_iterations=args.max_iterations,
        dream_updates=args.dream_updates,
        collect_episodes=args.collect_episodes,
        wm_fine_tune_steps=args.wm_steps,
        wm_lr=args.wm_lr,
        dream_lr=args.dream_lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
