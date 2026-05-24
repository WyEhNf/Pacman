"""Train the world model on collected gameplay data."""
import argparse
from pathlib import Path

import torch

from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.wm_trainer import WMTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the world model on collected replay data."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to replay_buffer.pt",
    )
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Where to save checkpoints. Default: same dir as data / world_model",
    )
    args = parser.parse_args()

    # --- Device detection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load replay buffer ---
    data_path = Path(args.data)
    print(f"Loading replay buffer from {data_path} ...")
    buffer = EpisodeReplayBuffer()
    buffer.load(str(data_path))
    print(f"  Episodes: {len(buffer)} | Total steps: {buffer.total_steps}")

    # --- Create world model ---
    world_model = WorldModel().to(device)
    param_count = sum(p.numel() for p in world_model.parameters())
    print(f"WorldModel parameters: {param_count:,}")

    # --- Save directory ---
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = data_path.parent / "world_model"
    print(f"Save directory: {save_dir}")

    # --- Train ---
    trainer = WMTrainer(
        world_model=world_model,
        replay_buffer=buffer,
        device=device,
        lr=args.lr,
        grad_clip=100.0,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    trainer.train(
        total_steps=args.total_steps,
        log_every=100,
        save_every=5000,
        save_dir=save_dir,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
