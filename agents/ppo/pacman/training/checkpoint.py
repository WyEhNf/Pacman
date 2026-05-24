# pacman/training/checkpoint.py
"""Checkpoint save/load for PPO training with optional RND state."""
from pathlib import Path
import torch


def save_checkpoint(
    path: Path,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_normalizer: dict,
    curriculum_phase: int,
    config: dict,
    is_best: bool = False,
    rnd_state: dict | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "update": update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward_normalizer": reward_normalizer,
        "curriculum_phase": curriculum_phase,
        "config": config,
    }
    if rnd_state is not None:
        data["rnd_state"] = rnd_state
    torch.save(data, path / f"update_{update}.pt")
    torch.save(data, path / "latest.pt")
    if is_best:
        torch.save(data, path / "best.pt")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    filename: str = "latest.pt",
) -> dict:
    data = torch.load(path / filename, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    return {
        "update": data["update"],
        "reward_normalizer": data.get("reward_normalizer", {}),
        "curriculum_phase": data.get("curriculum_phase", 0),
        "config": data.get("config", {}),
        "rnd_state": data.get("rnd_state", None),
    }
