"""Launch the dream viewer: side-by-side real vs imagined Pac-Man."""
import argparse
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.world_model.world_model import WorldModel
from pacman.viz.dream_viewer import DreamViewer


def _auto_device() -> torch.device:
    """Pick the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side dream viewer: real Pac-Man vs world model reconstruction",
    )
    parser.add_argument(
        "--world-model",
        type=str,
        required=True,
        help="Path to world_model_latest.pt checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: built-in default.yaml)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Pixel size of each maze tile (default: 16)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum visualization steps (default: 3000)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Detect device
    device = _auto_device()
    print(f"Using device: {device}")

    # Load world model
    model_path = Path(args.world_model)
    if not model_path.exists():
        raise FileNotFoundError(f"World model checkpoint not found: {model_path}")

    print(f"Loading world model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    world_model = WorldModel()
    if "model_state_dict" in checkpoint:
        world_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Support plain state_dict files
        world_model.load_state_dict(checkpoint)
    world_model.to(device)
    world_model.eval()
    print("World model loaded successfully.")

    # Launch viewer
    viewer = DreamViewer(
        world_model=world_model,
        config=config,
        device=device,
        tile_size=args.tile_size,
    )
    viewer.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
