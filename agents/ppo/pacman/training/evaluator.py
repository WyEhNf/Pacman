# pacman/training/evaluator.py
"""Greedy policy evaluation."""
import copy
import numpy as np
import torch

from ..env.pacman_env import PacmanEnv
from ..agents.networks import ActorCritic


def _run_episodes(env: PacmanEnv, network: ActorCritic, device: torch.device,
                  num_episodes: int) -> dict:
    scores, steps_list, ghosts_eaten, cleared = [], [], [], 0
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        ep_ghosts = 0
        while True:
            grid_t = torch.as_tensor(obs["grid"][None], device=device)
            scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
            mask_t = torch.as_tensor(env.get_legal_mask()[None], device=device)
            logits, _ = network(grid_t, scalars_t, mask_t)
            obs, _r, terminated, truncated, info = env.step(logits.argmax(dim=-1).item())
            for event in info.get("events", []):
                if event.startswith("eat_ghost"):
                    ep_ghosts += 1
            if terminated or truncated:
                scores.append(info["score"])
                steps_list.append(env.state.step_count)
                ghosts_eaten.append(ep_ghosts)
                if info.get("winner") == "pacman":
                    cleared += 1
                break
    return {
        "clear_rate": cleared / max(num_episodes, 1),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "mean_ghosts_eaten": float(np.mean(ghosts_eaten)) if ghosts_eaten else 0.0,
    }


class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.env_3life = PacmanEnv(config, difficulty=2)

        cfg_1life = copy.deepcopy(config)
        cfg_1life["game"]["lives"] = 1
        self.env_1life = PacmanEnv(cfg_1life, difficulty=2)

    @torch.no_grad()
    def evaluate(
        self,
        network: ActorCritic,
        num_episodes: int,
        device: torch.device,
    ) -> dict:
        network.eval()

        result_3 = _run_episodes(self.env_3life, network, device, num_episodes)
        result_1 = _run_episodes(self.env_1life, network, device, num_episodes)

        network.train()
        return {
            "clear_rate_3life": result_3["clear_rate"],
            "mean_score_3life": result_3["mean_score"],
            "mean_steps_3life": result_3["mean_steps"],
            "mean_ghosts_3life": result_3["mean_ghosts_eaten"],
            "clear_rate_1life": result_1["clear_rate"],
            "mean_score_1life": result_1["mean_score"],
            "mean_steps_1life": result_1["mean_steps"],
            "mean_ghosts_1life": result_1["mean_ghosts_eaten"],
        }
