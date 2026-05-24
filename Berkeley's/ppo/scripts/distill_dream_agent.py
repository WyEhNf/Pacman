"""Distill PPO agent behavior into a dream policy via behavioral cloning."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.world_model.world_model import WorldModel
from pacman.training.dream_trainer import DreamTrainer, DreamPolicy
from pacman.training.distill_ppo import collect_distillation_data, train_behavioral_cloning


def main():
    parser = argparse.ArgumentParser(description="Distill PPO into dream agent")
    parser.add_argument("--ppo-checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint (best.pt)")
    parser.add_argument("--world-model", type=str, required=True,
                        help="Path to world model checkpoint")
    parser.add_argument("--collect-episodes", type=int, default=500)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune", action="store_true",
                        help="Run imagination fine-tuning after BC")
    parser.add_argument("--ft-updates", type=int, default=200)
    parser.add_argument("--ft-lr", type=float, default=1e-5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    config = load_config()

    # Output directory
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.ppo_checkpoint).parent.parent / "distillation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load PPO agent ----
    print("\n[1/4] Loading PPO agent...")
    ppo_ckpt = torch.load(args.ppo_checkpoint, map_location=device, weights_only=False)
    ppo_config = ppo_ckpt["config"]
    net_cfg = ppo_config["network"]
    env_cfg = ppo_config["env"]

    ppo_network = ActorCritic(
        grid_channels=env_cfg["observation_channels"] * env_cfg["frame_stack"],
        num_scalars=env_cfg["num_scalar_features"],
        cnn_channels=net_cfg["cnn_channels"],
        cnn_kernels=net_cfg["cnn_kernels"],
        cnn_strides=net_cfg["cnn_strides"],
        shared_hidden=net_cfg["shared_hidden"],
        head_hidden=net_cfg["head_hidden"],
    ).to(device)
    ppo_network.load_state_dict(ppo_ckpt["model_state_dict"])
    ppo_network.eval()
    print(f"  Loaded PPO from {args.ppo_checkpoint}")

    # ---- Load World Model ----
    print("\n[2/4] Loading world model...")
    wm = WorldModel().to(device)
    wm_ckpt = torch.load(args.world_model, map_location=device, weights_only=True)
    wm.load_state_dict(wm_ckpt["model_state_dict"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"  Loaded WM from {args.world_model}")

    # ---- Phase 1: Collect distillation data ----
    print(f"\n[3/4] Collecting distillation data ({args.collect_episodes} episodes)...")
    t0 = time.time()
    data = collect_distillation_data(
        ppo_network=ppo_network,
        world_model=wm,
        config=config,
        device=device,
        num_episodes=args.collect_episodes,
        difficulty=2,
    )
    data_path = save_dir / "distillation_data.pt"
    torch.save(data, data_path)
    print(f"  Saved {data['latents'].shape[0]} pairs to {data_path} ({time.time() - t0:.1f}s)")

    # ---- Phase 2: Behavioral Cloning ----
    print(f"\n[4/4] Training behavioral cloning ({args.bc_epochs} epochs)...")
    policy = DreamPolicy(latent_dim=wm.rssm.latent_dim).to(device)

    bc_result = train_behavioral_cloning(
        policy=policy,
        latents=data["latents"],
        actions=data["actions"],
        device=device,
        epochs=args.bc_epochs,
        lr=args.bc_lr,
    )
    print(f"  Val accuracy: {bc_result['val_accuracy']:.1%}")

    # Save distilled policy
    distilled_path = save_dir / "distilled_policy.pt"
    torch.save({"policy_state_dict": policy.state_dict()}, distilled_path)
    print(f"  Saved distilled policy to {distilled_path}")

    # ---- Evaluate distilled policy ----
    print(f"\nEvaluating distilled policy ({args.eval_episodes} episodes)...")
    trainer = DreamTrainer(
        world_model=wm,
        config=config,
        device=device,
        imagination_horizon=5,
        num_imaginations=512,
    )
    # Replace the random policy with our distilled one
    trainer.policy = policy
    eval_result = trainer._evaluate_in_real_env(args.eval_episodes)
    print(
        f"  Distilled agent: mean_score={eval_result['mean_score']:.1f} "
        f"level_clear={eval_result['level_clear_rate']:.1%}"
    )

    # ---- Phase 3: Optional fine-tuning ----
    if args.fine_tune and eval_result["mean_score"] > 0:
        print(f"\nFine-tuning with imagination PPO ({args.ft_updates} updates)...")
        ft_trainer = DreamTrainer(
            world_model=wm,
            config=config,
            device=device,
            imagination_horizon=5,
            num_imaginations=512,
            lr=args.ft_lr,
            entropy_coef_start=0.1,
            entropy_coef_end=0.01,
            latent_noise=0.15,
        )
        # Load distilled actor weights, keep fresh critic
        ft_trainer.policy.actor.load_state_dict(policy.actor.state_dict())

        ft_result = ft_trainer.train(
            total_updates=args.ft_updates,
            eval_every=25,
            eval_episodes=args.eval_episodes,
            save_dir=save_dir / "fine_tuned",
            patience=100,
        )
        print(f"  Fine-tuned best score: {ft_result['best_score']:.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
