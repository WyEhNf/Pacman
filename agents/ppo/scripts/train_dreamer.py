"""Train a dream agent (imagination PPO) using a pretrained world model."""
import argparse
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.training.dream_trainer import DreamTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train a dream agent via imagination PPO inside a learned world model."
    )
    parser.add_argument(
        "--world-model", type=str, required=True,
        help="Path to world_model_latest.pt checkpoint.",
    )
    parser.add_argument("--total-updates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-imaginations", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Where to save dream agent checkpoints. Default: world model dir / dream_agent",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    # --- Config ---
    config = load_config(args.config)

    # --- Device detection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load world model ---
    wm_path = Path(args.world_model)
    print(f"Loading world model from {wm_path} ...")
    world_model = WorldModel()
    checkpoint = torch.load(wm_path, map_location=device, weights_only=True)
    world_model.load_state_dict(checkpoint["model_state_dict"])
    print("  World model loaded successfully.")

    # --- Save directory ---
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = wm_path.parent / "dream_agent"
    print(f"Save directory: {save_dir}")

    # --- Create trainer and run ---
    trainer = DreamTrainer(
        world_model=world_model,
        config=config,
        device=device,
        imagination_horizon=args.horizon,
        num_imaginations=args.num_imaginations,
        lr=args.lr,
    )

    policy_params = sum(p.numel() for p in trainer.policy.parameters())
    print(f"DreamPolicy parameters: {policy_params:,}")

    try:
        trainer.train(
            total_updates=args.total_updates,
            save_dir=save_dir,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer._save(save_dir, 0, is_best=False)
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
