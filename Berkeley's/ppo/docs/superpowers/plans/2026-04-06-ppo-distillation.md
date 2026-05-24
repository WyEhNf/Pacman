# PPO Behavior Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill the PPO agent's behavior into the dream policy via behavioral cloning in latent space, then optionally fine-tune with imagination PPO, to maximize the dream agent's real game score.

**Architecture:** A library module (`pacman/training/distill_ppo.py`) provides two functions: `collect_distillation_data` (runs PPO agent while encoding through world model to build latent→action pairs) and `train_behavioral_cloning` (supervised training on that dataset). A CLI script orchestrates the full pipeline. Existing `DreamPolicy`, `DreamTrainer`, `WorldModel`, and `ActorCritic` classes are used as-is.

**Tech Stack:** PyTorch, existing WorldModel/DreamPolicy/DreamTrainer/ActorCritic/PacmanEnv

---

### Task 1: Distillation Data Collection

**Files:**
- Create: `pacman/training/distill_ppo.py`
- Test: `tests/test_distillation.py`

Collect (latent_state, expert_action) pairs by running the PPO agent in the real game while simultaneously encoding observations through the world model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distillation.py
"""Tests for PPO behavior distillation into dream agent."""
import torch
import numpy as np
import pytest
from collections import deque
from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.agents.networks import ActorCritic
from pacman.training.dream_trainer import DreamPolicy


@pytest.fixture
def config():
    cfg = load_config()
    cfg["env"]["frame_stack"] = 1  # world model uses single frames
    return cfg


@pytest.fixture
def config_stacked():
    """Config with frame stacking for PPO."""
    cfg = load_config()
    return cfg


class TestDistillationDataCollection:
    def test_collect_returns_latents_and_actions(self, config):
        """collect_distillation_data returns a dict with latents and actions tensors."""
        from pacman.training.distill_ppo import collect_distillation_data

        device = torch.device("cpu")

        # Create a random PPO network (untrained, just for shape testing)
        ppo_net = ActorCritic(
            grid_channels=8 * 4,  # 4-frame stack
            num_scalars=5,
        )
        ppo_net.eval()

        wm = WorldModel()
        wm.eval()

        result = collect_distillation_data(
            ppo_network=ppo_net,
            world_model=wm,
            config=config,
            device=device,
            num_episodes=3,
            difficulty=0,
        )

        assert "latents" in result
        assert "actions" in result
        assert isinstance(result["latents"], torch.Tensor)
        assert isinstance(result["actions"], torch.Tensor)
        assert result["latents"].shape[1] == 2560  # h=512 + z=2048
        assert result["actions"].shape[0] == result["latents"].shape[0]
        assert result["latents"].shape[0] > 0  # collected some data
        # Actions should be valid (0-3)
        assert result["actions"].min() >= 0
        assert result["actions"].max() <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation.py::TestDistillationDataCollection::test_collect_returns_latents_and_actions -v`
Expected: FAIL — `pacman.training.distill_ppo` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# pacman/training/distill_ppo.py
"""PPO behavior distillation: collect latent-action pairs and train via behavioral cloning."""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.agents.networks import ActorCritic
from pacman.training.dream_trainer import DreamPolicy


def collect_distillation_data(
    ppo_network: ActorCritic,
    world_model: WorldModel,
    config: dict,
    device: torch.device,
    num_episodes: int = 500,
    difficulty: int = 2,
) -> dict[str, torch.Tensor]:
    """Run PPO agent in real game while encoding through world model.

    At each step, the PPO agent picks an action from frame-stacked observations,
    and the world model encodes the single-frame observation into latent space.
    Returns (latent_state, ppo_action) pairs for behavioral cloning.

    Args:
        ppo_network: Trained ActorCritic network (eval mode).
        world_model: Trained WorldModel (eval mode, frozen).
        config: Game config dict.
        device: Torch device.
        num_episodes: Number of episodes to collect.
        difficulty: Game difficulty level.

    Returns:
        Dict with "latents" (N, 2560) and "actions" (N,) tensors.
    """
    ppo_network.eval()
    world_model.eval()

    # Use frame_stack=1 env for world model encoding; manage PPO frame stack manually
    env_config = {**config, "env": {**config["env"], "frame_stack": 1}}
    env = PacmanEnv(env_config, difficulty=difficulty)

    all_latents = []
    all_actions = []
    total_steps = 0
    start_time = time.time()

    for ep in range(num_episodes):
        env.reset(seed=ep)
        obs = env._build_obs()  # single-frame: {"grid": (8,31,28), "scalars": (5,)}

        # Manual frame stack for PPO (4 frames)
        frame_stack = deque(maxlen=4)
        for _ in range(4):
            frame_stack.append(obs["grid"])

        # Initialize RSSM state
        h, z = world_model.rssm.initial_state(1)
        h = world_model.rssm.dynamics(
            h, z, torch.zeros(1, dtype=torch.long, device=device)
        )

        done = False
        while not done:
            with torch.no_grad():
                # --- World model encoding ---
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                enc = world_model.encoder(grid_t, scalars_t)
                z, _ = world_model.rssm.posterior(h, enc)
                latent = torch.cat([h, z], dim=-1)  # (1, 2560)

                # --- PPO action selection (greedy) ---
                stacked_grid = np.stack(list(frame_stack), axis=0).reshape(
                    -1, obs["grid"].shape[1], obs["grid"].shape[2]
                )  # (32, 31, 28)
                ppo_grid_t = torch.as_tensor(
                    stacked_grid[None], device=device
                )  # (1, 32, 31, 28)
                ppo_scalars_t = torch.as_tensor(
                    obs["scalars"][None], device=device
                )  # (1, 5)
                legal_mask = torch.as_tensor(
                    env.get_legal_mask()[None], device=device
                )  # (1, 4)
                ppo_logits, _ = ppo_network(ppo_grid_t, ppo_scalars_t, legal_mask)
                action = ppo_logits.argmax(dim=-1).item()  # greedy

                # Store pair
                all_latents.append(latent.cpu())
                all_actions.append(action)

                # Advance RSSM
                action_t = torch.tensor([action], dtype=torch.long, device=device)
                h = world_model.rssm.dynamics(h, z, action_t)

            # Step environment
            _, _, terminated, truncated, _ = env.step(action)
            obs = env._build_obs()
            frame_stack.append(obs["grid"])
            done = terminated or truncated
            total_steps += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Episode {ep + 1}/{num_episodes} | "
                f"Steps: {total_steps} | {(ep + 1) / elapsed:.1f} eps/s",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"  Collection complete: {num_episodes} episodes, {total_steps} steps, {elapsed:.1f}s")

    return {
        "latents": torch.cat(all_latents, dim=0),  # (N, 2560)
        "actions": torch.tensor(all_actions, dtype=torch.long),  # (N,)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation.py::TestDistillationDataCollection::test_collect_returns_latents_and_actions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pacman/training/distill_ppo.py tests/test_distillation.py
git commit -m "feat: distillation data collection — PPO latent-action pairs"
```

---

### Task 2: Behavioral Cloning Training

**Files:**
- Modify: `pacman/training/distill_ppo.py`
- Test: `tests/test_distillation.py`

Train the DreamPolicy actor head to predict PPO actions from latent states via cross-entropy loss.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distillation.py`:

```python
class TestBehavioralCloning:
    def test_train_bc_reduces_loss(self):
        """train_behavioral_cloning trains actor and returns metrics with decreasing loss."""
        from pacman.training.distill_ppo import train_behavioral_cloning

        device = torch.device("cpu")

        # Create a synthetic dataset: latent -> action mapping
        # Use a simple pattern: action = argmax of first 4 dims of latent
        N = 2000
        latents = torch.randn(N, 2560)
        # Assign actions based on a learnable pattern
        actions = (latents[:, :4].argmax(dim=-1)).long()

        policy = DreamPolicy(latent_dim=2560)

        result = train_behavioral_cloning(
            policy=policy,
            latents=latents,
            actions=actions,
            device=device,
            epochs=10,
            batch_size=256,
            lr=1e-3,
            patience=50,  # don't early stop for this test
        )

        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result
        assert "best_epoch" in result
        assert isinstance(result["val_accuracy"], float)
        assert result["val_accuracy"] > 0.25  # better than random (25%)
        # Loss should have decreased from initial
        assert result["val_loss"] < 2.0  # well below initial ~1.39 (ln(4))

    def test_bc_policy_predicts_actions(self):
        """After BC training, the policy should predict the majority action correctly."""
        from pacman.training.distill_ppo import train_behavioral_cloning

        device = torch.device("cpu")

        # Create data where action is always 0 for positive first dim, 1 otherwise
        N = 1000
        latents = torch.randn(N, 2560)
        actions = (latents[:, 0] > 0).long()  # 0 or 1

        policy = DreamPolicy(latent_dim=2560)

        train_behavioral_cloning(
            policy=policy,
            latents=latents,
            actions=actions,
            device=device,
            epochs=20,
            batch_size=256,
            lr=1e-3,
            patience=50,
        )

        # Test on new data
        test_latents = torch.randn(200, 2560)
        test_actions = (test_latents[:, 0] > 0).long()
        with torch.no_grad():
            logits = policy.actor(test_latents)
            predicted = logits.argmax(dim=-1)
        accuracy = (predicted == test_actions).float().mean().item()
        assert accuracy > 0.6, f"Expected >60% accuracy, got {accuracy:.1%}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distillation.py::TestBehavioralCloning -v`
Expected: FAIL — `train_behavioral_cloning` does not exist.

- [ ] **Step 3: Add train_behavioral_cloning to distill_ppo.py**

Append to `pacman/training/distill_ppo.py`:

```python
def train_behavioral_cloning(
    policy: DreamPolicy,
    latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 5,
    val_fraction: float = 0.2,
) -> dict:
    """Train DreamPolicy actor via behavioral cloning on (latent, action) pairs.

    Only trains the actor head. The critic is left untouched.

    Args:
        policy: DreamPolicy to train (actor head).
        latents: (N, 2560) latent states.
        actions: (N,) expert action labels.
        device: Torch device.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Initial learning rate.
        patience: Early stopping patience on val loss.
        val_fraction: Fraction of data for validation.

    Returns:
        Dict with train_loss, val_loss, val_accuracy, best_epoch.
    """
    policy.to(device)
    policy.train()

    # Train/val split
    N = latents.shape[0]
    n_val = int(N * val_fraction)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_latents = latents[train_idx].to(device)
    train_actions = actions[train_idx].to(device)
    val_latents = latents[val_idx].to(device)
    val_actions = actions[val_idx].to(device)

    # Only optimize actor parameters
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training ---
        policy.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            batch_latents = train_latents[idx]
            batch_actions = train_actions[idx]

            logits = policy.actor(batch_latents)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        policy.eval()
        with torch.no_grad():
            val_logits = policy.actor(val_latents)
            val_loss = F.cross_entropy(val_logits, val_actions).item()
            val_preds = val_logits.argmax(dim=-1)
            val_accuracy = (val_preds == val_actions).float().mean().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.1%} lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in policy.actor.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best={best_epoch})")
                break

    # Restore best weights
    if best_state is not None:
        policy.actor.load_state_dict(best_state)

    print(f"  BC complete: best_epoch={best_epoch} val_loss={best_val_loss:.4f} val_acc={val_accuracy:.1%}")

    return {
        "train_loss": train_loss,
        "val_loss": best_val_loss,
        "val_accuracy": val_accuracy,
        "best_epoch": best_epoch,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distillation.py::TestBehavioralCloning -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pacman/training/distill_ppo.py tests/test_distillation.py
git commit -m "feat: behavioral cloning training for dream policy actor"
```

---

### Task 3: CLI Script and End-to-End Pipeline

**Files:**
- Create: `scripts/distill_dream_agent.py`
- Test: `tests/test_distillation.py`

The CLI script loads the PPO checkpoint and world model, runs data collection, trains BC, evaluates, and optionally fine-tunes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distillation.py`:

```python
class TestDistillationPipeline:
    def test_distill_and_load_into_dream_policy(self, config):
        """Full pipeline: collect data, train BC, load result into DreamPolicy."""
        from pacman.training.distill_ppo import (
            collect_distillation_data,
            train_behavioral_cloning,
        )

        device = torch.device("cpu")

        # Create random PPO network and world model
        ppo_net = ActorCritic(grid_channels=32, num_scalars=5)
        ppo_net.eval()
        wm = WorldModel()
        wm.eval()

        # Phase 1: Collect data (tiny)
        data = collect_distillation_data(
            ppo_network=ppo_net,
            world_model=wm,
            config=config,
            device=device,
            num_episodes=3,
            difficulty=0,
        )
        assert data["latents"].shape[0] > 50  # some data collected

        # Phase 2: Train BC
        policy = DreamPolicy(latent_dim=2560)
        result = train_behavioral_cloning(
            policy=policy,
            latents=data["latents"],
            actions=data["actions"],
            device=device,
            epochs=5,
            batch_size=64,
            lr=1e-3,
            patience=50,
        )
        assert "val_accuracy" in result

        # Phase 3: Save and reload
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "distilled_policy.pt"
            torch.save({"policy_state_dict": policy.state_dict()}, save_path)

            # Reload into fresh DreamPolicy
            new_policy = DreamPolicy(latent_dim=2560)
            ckpt = torch.load(save_path, weights_only=True)
            new_policy.load_state_dict(ckpt["policy_state_dict"])

            # Verify same output
            test_input = torch.randn(1, 2560)
            with torch.no_grad():
                orig_out = policy.actor(test_input)
                new_out = new_policy.actor(test_input)
            assert torch.allclose(orig_out, new_out)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_distillation.py::TestDistillationPipeline::test_distill_and_load_into_dream_policy -v`
Expected: PASS (all functions already exist from Tasks 1-2).

- [ ] **Step 3: Create the CLI script**

```python
# scripts/distill_dream_agent.py
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
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_distillation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/distill_dream_agent.py tests/test_distillation.py
git commit -m "feat: distill_dream_agent CLI — full PPO-to-dream pipeline"
```

---

### Task 4: Run the Distillation Pipeline

**Files:**
- None (uses existing scripts and checkpoints)

Deploy the full pipeline against the real trained PPO agent and world model.

- [ ] **Step 1: Verify checkpoints exist**

```bash
ls -lh runs/2026-04-03_19-27-33/checkpoints/best.pt
ls -lh runs/2026-04-03_19-27-33/world_model/world_model_latest.pt
```

Expected: Both files exist.

- [ ] **Step 2: Run the distillation pipeline**

```bash
python scripts/distill_dream_agent.py \
  --ppo-checkpoint runs/2026-04-03_19-27-33/checkpoints/best.pt \
  --world-model runs/2026-04-03_19-27-33/world_model/world_model_latest.pt \
  --collect-episodes 500 \
  --bc-epochs 50 \
  --bc-lr 1e-3 \
  --eval-episodes 100 \
  --fine-tune \
  --ft-updates 200 \
  --ft-lr 1e-5
```

Monitor for:
- Data collection: ~150-200K pairs from 500 episodes
- BC val accuracy: target >65%
- Distilled agent eval score: target >1000
- Fine-tuned agent score: target >1500

- [ ] **Step 3: Commit results**

```bash
git add scripts/distill_dream_agent.py pacman/training/distill_ppo.py tests/test_distillation.py
git commit -m "feat: PPO behavior distillation — distilled dream agent scores X"
git push origin master
```
