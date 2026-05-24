# PPO Behavior Distillation into Dream Agent

## Goal

Maximize the dream agent's real game score by distilling the PPO agent's behavior into the dream policy via behavioral cloning in latent space, then optionally fine-tuning with a short imagination PPO burst.

## Problem Statement

The dream agent peaks at ~675 in real eval and rapidly overfits to world model imperfections. The online dream loop (iterative WM fine-tuning) converged after 2 iterations without improvement. The fundamental issue: imagination-only RL gives the agent thousands of gradient updates to exploit world model errors.

Instead of trying to fix the world model, we sidestep the problem: use supervised learning to teach the dream policy what a good player (the PPO agent) does at each latent state. This avoids RL-in-imagination entirely for the initial policy, and limits imagination RL to a short fine-tuning burst.

## Architecture

### Phase 1: Distillation Data Collection

Run the PPO agent in the real game while simultaneously encoding observations through the world model to build a dataset of (latent_state, expert_action) pairs.

**Flow per episode:**
1. Reset env, initialize RSSM state `(h, z)` with `wm.rssm.initial_state(1)`
2. Advance RSSM with a dummy action: `h = wm.rssm.dynamics(h, z, zeros)`
3. At each step:
   - Get single-frame obs via `env._build_obs()` for the world model
   - Encode: `enc = wm.encoder(grid, scalars)` then `z, _ = wm.rssm.posterior(h, enc)`
   - Form latent: `[h; z]` (2560 dims)
   - Get PPO action from the frame-stacked observation (32ch grid + 5 scalars + legal mask)
   - Use greedy action (argmax of PPO logits) for cleaner supervision signal
   - Store `(latent, action)` pair
   - Advance RSSM: `h = wm.rssm.dynamics(h, z, action)`
   - Step env, update frame stack buffer
4. Repeat for all episodes

**Frame stacking for PPO:** The PPO agent requires 4-frame stacked observations (32 channels). Since we run a single `PacmanEnv` (not VecEnv), we maintain a manual frame stack buffer (deque of 4 grids). On reset, all 4 slots are filled with the initial frame.

**Parameters:**
- 500 episodes at difficulty 2 (full ghost AI)
- Expected yield: ~150-200K (latent, action) pairs
- PPO checkpoint: `runs/2026-04-03_19-27-33/checkpoints/best.pt`
- World model: `runs/2026-04-03_19-27-33/world_model/world_model_latest.pt`

**Output:** A single `.pt` file containing `{"latents": Tensor(N, 2560), "actions": Tensor(N,)}`

### Phase 2: Behavioral Cloning

Train the DreamPolicy actor to predict the PPO agent's actions from latent states.

- **Model:** Existing `DreamPolicy` class (actor head only; critic is not trained here)
- **Loss:** Cross-entropy between `DreamPolicy.actor(latent)` logits and the PPO action label
- **Dataset:** 80/20 train/val split of the collected (latent, action) pairs
- **Batch size:** 512
- **Learning rate:** 1e-3 with cosine annealing to 1e-5
- **Epochs:** 50 max, early stop on val loss with patience 5
- **Diagnostic metric:** Top-1 action accuracy on val set (random baseline = 25%)

**Output:** Saved DreamPolicy checkpoint with BC-trained actor weights: `distilled_policy.pt`

### Phase 3: Dream Fine-tune (Optional)

Load the distilled policy into `DreamTrainer` and run a short imagination PPO burst.

**Why this can exceed PPO performance:** The PPO agent sees 4 stacked frames (short-term temporal info). The dream agent's latent state `[h; z]` encodes the entire episode history through the GRU — strictly more information is available for decision-making.

**Training setup:**
- Load distilled DreamPolicy weights (actor only, critic initializes fresh)
- DreamTrainer with conservative settings:
  - `total_updates=200` (not 3000)
  - `eval_every=25`, `eval_episodes=100`
  - `patience=100`
  - `lr=1e-5` (fine-tuning LR, not training-from-scratch LR)
  - `imagination_horizon=5` (shorter to reduce prior drift)
  - `entropy_coef_start=0.1`, `entropy_coef_end=0.01`
  - `latent_noise=0.15`
- Starting states: during Phase 1 data collection, save every 50th (grid, scalars) pair. These become the starting observations for `DreamTrainer._get_starting_states()` override, ensuring the agent imagines from states the PPO agent actually visits.

**Decision gate:** Evaluate the distilled policy (BC-only) first. If it already scores >1500, proceed with fine-tuning. If it scores <500, the latent representations may not preserve enough info, and fine-tuning won't help.

### Phase 4: Evaluation

Compare all agents on 100 real episodes at difficulty 2:
1. Random dream policy (baseline)
2. Best existing dream agent (675, from online loop iter 0)
3. Distilled dream policy (BC only)
4. Fine-tuned dream policy (if Phase 3 is run)
5. PPO agent (the teacher, upper reference)

Report: mean score, std, level clear rate.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `pacman/training/distill_ppo.py` | Library: data collection + behavioral cloning training |
| `scripts/distill_dream_agent.py` | CLI entry point for the full distillation pipeline |
| `tests/test_distillation.py` | Tests for data collection, BC training, policy loading |

### Unchanged Files

All existing files remain untouched. `DreamPolicy`, `DreamTrainer`, `WorldModel`, `ActorCritic`, `PPO`, `PacmanEnv` are used as-is. The distilled weights are loaded into `DreamPolicy` via `load_state_dict` on the actor sub-module.

## Loading the PPO Agent

The PPO checkpoint (`best.pt`) contains:
- `model_state_dict`: weights for `ActorCritic(grid_channels=32, num_scalars=5, ...)`
- `config`: the training config dict (includes network architecture)

To reconstruct:
```
ckpt = torch.load(path, weights_only=False)
config = ckpt["config"]
net_cfg = config["network"]
network = ActorCritic(
    grid_channels=config["env"]["observation_channels"] * config["env"]["frame_stack"],
    num_scalars=config["env"]["num_scalar_features"],
    cnn_channels=net_cfg["cnn_channels"],
    ...
)
network.load_state_dict(ckpt["model_state_dict"])
```

## Expected Results

| Agent | Expected Score | Rationale |
|-------|---------------|-----------|
| Random dream policy | ~100-200 | No learning |
| Best existing dream (iter 0) | ~675 | Current ceiling |
| Distilled (BC only) | ~1000-2000 | If latent preserves PPO-relevant info |
| Fine-tuned | ~1500-3000+ | Exploiting richer latent representation |
| PPO agent | ~2000-3000 | The teacher |

## Compute Budget

- Data collection: ~20 min (500 episodes, single env)
- Behavioral cloning: ~5 min (50 epochs on 200K samples, GPU/MPS)
- Dream fine-tune: ~10 min (200 updates, small horizon)
- Evaluation: ~10 min (100 episodes x 5 agents)
- **Total: ~45 min**

## Success Criteria

1. **Primary:** Distilled dream agent scores >1000 in 100-episode real eval (beating the 675 ceiling)
2. **Stretch:** Fine-tuned dream agent matches or exceeds PPO agent score
3. **Diagnostic:** Behavioral cloning val accuracy >65%
