"""Continue training from checkpoint with modified config (death=-50)."""
import sys, os, time, numpy as np, torch, yaml
from pathlib import Path

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.agents.ppo import PPO
from pacman.agents.rollout import RolloutBuffer
from pacman.env.vec_env import VecEnv
from pacman.training.evaluator import Evaluator

# Load base config, modify death penalty
config = load_config()
config["rewards"]["death"] = -50.0
config["training"]["total_updates"] = 12000

device = torch.device("cuda")
env_cfg = config["env"]; net_cfg = config["network"]; train_cfg = config["training"]
ppo_cfg = config["ppo"]; N = env_cfg["num_envs"]; T = ppo_cfg["rollout_steps"]
fs = env_cfg.get("frame_stack", 4)

# Build network
network = ActorCritic(
    grid_channels=env_cfg["observation_channels"] * fs,
    num_scalars=env_cfg.get("num_scalar_features", 5),
    cnn_channels=net_cfg["cnn_channels"], cnn_kernels=net_cfg["cnn_kernels"],
    cnn_strides=net_cfg["cnn_strides"], shared_hidden=net_cfg["shared_hidden"],
    head_hidden=net_cfg["head_hidden"],
).to(device)

# Load checkpoint
ckpt_path = "runs/2026-05-24_00-23-37/checkpoints/update_7999.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
network.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint: update {ckpt['update']}")

# PPO
ppo = PPO(network, device, ppo_cfg)
ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

# Env
vec_env = VecEnv(N, config, difficulty=2)
rollout = RolloutBuffer(N, T, env_cfg["observation_channels"], fs)
evaluator = Evaluator(config)

class RMS:
    def __init__(self): self.mean=0.0; self.var=1.0; self.count=1e-4
    def update(self, x):
        bm,bv,bc=x.mean(),x.var(),x.shape[0]
        d=bm-self.mean; t=self.count+bc
        self.mean+=d*bc/t; self.var=(self.var*self.count+bv*bc+d*d*self.count*bc/t)/t
        self.count=t
    def normalize(self,x): return np.clip((x-self.mean)/(np.sqrt(self.var)+1e-8),-10,10)

rew_norm = RMS()
if "reward_normalizer" in ckpt:
    rew_norm.mean=ckpt["reward_normalizer"]["mean"]
    rew_norm.var=ckpt["reward_normalizer"]["var"]
    rew_norm.count=ckpt["reward_normalizer"]["count"]

start_update = ckpt["update"] + 1
total = 12000
best_clear = ckpt.get("clear_rate", 0.9)
curriculum_phase = ckpt.get("curriculum_phase", 2)
print(f"Continue from {start_update} to {total}, best_clear={best_clear:.1%}, death_penalty=-50\n")

obs = vec_env.reset(seed=42)

for update in range(start_update, total):
    t0 = time.time()

    ppo.anneal_lr(update, total)
    ent_coef = 0.15 + (0.01 - 0.15) * (update / total) ** 0.7
    ppo.set_entropy_coef(ent_coef)

    # Set curriculum phase (phase 2 = hardest, already at update 8000)
    vec_env.set_difficulty(2)

    # Collect rollout
    rollout.reset()
    for _step in range(T):
        masks = vec_env.get_legal_masks()
        actions, log_probs, values = ppo.select_action(obs["grid"], obs["scalars"], masks)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        rollout.add(obs, actions, log_probs, values, rewards, dones)
        obs = next_obs

    # Normalize rewards
    all_rewards = rollout.rewards.flatten()
    rew_norm.update(all_rewards[all_rewards != 0])
    rollout.rewards = rew_norm.normalize(rollout.rewards)

    # PPO update
    ppo.update(rollout, update)

    # Eval
    if update % 50 == 0:
        result = evaluator.evaluate(network, 20, device)
        fps = (N * T * 50) / max(time.time() - t0, 0.01)
        print(f"[Update {update}] clear={result['level_clear_rate']:.1%} "
              f"score={result['mean_score']:.0f} fps={fps:.0f}")
        if result["level_clear_rate"] > best_clear:
            best_clear = result["level_clear_rate"]
            torch.save({"update":update, "model_state_dict":network.state_dict(),
                        "optimizer_state_dict":ppo.optimizer.state_dict(),
                        "clear_rate":best_clear, "config":config},
                       f"runs/2026-05-24_00-23-37/checkpoints/best.pt")

    # Checkpoint
    if update % 100 == 0:
        torch.save({"update":update, "model_state_dict":network.state_dict(),
                    "optimizer_state_dict":ppo.optimizer.state_dict(),
                    "reward_normalizer":{"mean":rew_norm.mean,"var":rew_norm.var,"count":rew_norm.count},
                    "clear_rate":best_clear, "config":config},
                   f"runs/2026-05-24_00-23-37/checkpoints/update_{update}.pt")
        print(f"  [Saved update_{update}.pt]")

print(f"\nDone! Best clear: {best_clear:.1%}")
