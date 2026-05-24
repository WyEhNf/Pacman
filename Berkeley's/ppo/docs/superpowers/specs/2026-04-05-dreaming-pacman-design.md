# Dreaming Pac-Man: World Model for Imagination-Based Learning

**Date:** 2026-04-05
**Status:** Design approved
**Timeline:** 1-2 weeks
**Hardware:** MacBook M5 Pro (15-core CPU, 16-core GPU, 24GB RAM)

## Overview

Build a latent world model (RSSM) that learns to simulate Pac-Man entirely from gameplay data, then train an RL agent purely inside the model's "imagination" — no real environment interaction needed. The project demonstrates the core world model paradigm that underpins Dreamer (DeepMind), DIAMOND (NeurIPS 2024), and Genie 3 on consumer hardware.

### One-sentence pitch

"A neural network that learns to dream Pac-Man, then an agent that learns to play by dreaming."

## Architecture

### Three-phase pipeline

```
Phase A: Data Collection
  Trained PPO agent → plays 5000 episodes → replay buffer (~15M frames)

Phase B: World Model Training
  Replay buffer → train RSSM → learned simulator of Pac-Man

Phase C: Imagination Training
  RSSM generates imagined rollouts → PPO trains entirely in dreams → deploy to real game
```

### Phase A — Data Collection

Use the trained PPO agent (update 8000, 95-100% clear rate) to collect gameplay data.

**Per timestep:**
- `obs_grid`: (8, 31, 28) float32 — single-frame 8-channel observation (NOT frame-stacked; the RSSM's GRU replaces frame stacking with learned temporal memory)
- `obs_scalars`: (5,) float32 — scalar features (lives, score, power timer, mode, step)
- `action`: int [0-3] — UP/LEFT/DOWN/RIGHT
- `reward`: float32 — shaped reward
- `done`: bool — episode termination

**Note on frame stacking:** The current PPO agent uses 4-frame stacking (32 channels) to give the CNN temporal context. The world model does NOT need frame stacking — the GRU hidden state `h` serves the same purpose, maintaining temporal memory across steps. Data collection stores single frames.

**Collection parameters:**
- Episodes: 5000
- Estimated frames: ~15M (avg ~3000 steps/episode)
- Storage: sequential episodes (not shuffled — temporal order matters)
- Estimated time: ~30 minutes using vectorized engine on 15 CPU cores
- Storage format: chunked `.pt` files, ~2GB total

### Phase B — World Model (RSSM)

The Recurrent State Space Model learns a compressed simulation of the game in latent space.

**Latent state has two parts:**
- Deterministic state `h` (GRU hidden): captures long-term memory — mode timers, ghost patterns, game phase
- Stochastic state `z` (categorical): captures the specific current situation — exact positions, immediate threats

**Five neural network components:**

| Component | Architecture | Input → Output | Purpose |
|-----------|-------------|----------------|---------|
| Encoder | 4-layer CNN [64,128,256,256] + MLP | observation → z_posterior | Compress real frame into latent |
| Dynamics | GRU (512 hidden) | (h_t, z_t, action) → h_{t+1} | Evolve deterministic state forward |
| Prior | MLP (512 hidden, 2 layers) | h_{t+1} → z_prior | Predict stochastic state without seeing real frame |
| Decoder | MLP + 4-layer transposed CNN | (h, z) → reconstructed obs | Visualize what the model "sees" |
| Reward head | MLP (512 hidden, 3 layers) | (h, z) → predicted reward | Predict reward for imagination training |
| Continue head | MLP (512 hidden, 3 layers) | (h, z) → p(not done) | Predict episode continuation probability |

**Sizing:**

| Parameter | Value |
|-----------|-------|
| GRU hidden dim (h) | 512 |
| Stochastic classes | 32 |
| Stochastic categoricals | 64 |
| Stochastic state dim (z) | 32 x 64 = 2048 |
| Full latent dim (h + z) | 2560 |
| CNN encoder channels | [64, 128, 256, 256] |
| CNN encoder kernels | [4, 4, 4, 4] |
| CNN encoder strides | [2, 2, 2, 1] |
| Decoder | mirror of encoder with transposed convolutions |
| Total parameters | ~8-10M |

**Training loss:**
```
L = reconstruction_loss + reward_loss + continue_loss + beta * KL(posterior || prior)
```

- Reconstruction: MSE on decoded observation vs. real observation
- Reward: MSE on predicted vs. actual reward (symlog transform for scale)
- Continue: binary cross-entropy on termination prediction
- KL: forces the prior (which doesn't see real frames) to match the posterior (which does)
- beta: 1.0 (free nats: 1.0 — allows some KL without penalty to avoid posterior collapse)

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 128 (16 sequences x 8 parallel) |
| Sequence length | 50 |
| Training steps | 100,000 |
| KL balancing (beta) | 1.0 |
| Free nats | 1.0 |
| Gradient clipping | 100.0 (global norm) |
| Device | MPS (16-core GPU) |
| Estimated RAM usage | ~18-20GB |
| Estimated training time | 6-10 hours |

**Scalar features handling:**
The 5 scalar features (lives, score, power timer, mode, step) are embedded via a small MLP (5 → 64) and concatenated with the CNN encoder output before the final encoding layer. The decoder also produces scalar predictions alongside the grid reconstruction.

### Phase C — Imagination Training

Once the world model is trained, use it as a virtual environment for PPO.

**Imagination rollout (pseudocode):**
```python
h, z = world_model.encode(real_starting_observation)
for t in range(imagination_horizon):
    action = policy_network(h, z)
    h = world_model.dynamics(h, z, action)
    z = world_model.prior(h)  # sample from learned prior
    reward = world_model.reward_head(h, z)
    cont = world_model.continue_head(h, z)
    store(h, z, action, reward, cont)
# Compute GAE and PPO update on imagined trajectories
```

**Key parameters:**

| Parameter | Value |
|-----------|-------|
| Imagination horizon | 15 steps |
| Parallel imaginations | 512 |
| Policy network | 2-layer MLP (2560 → 512 → 256 → 4 actions) |
| Value network | 2-layer MLP (2560 → 512 → 256 → 1 value) |
| PPO epochs | 4 |
| PPO clip epsilon | 0.2 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | 1e-4 |
| Entropy coefficient | 0.01 |
| Total imagination updates | 5000 |
| Real-env validation every | 100 updates (20 episodes) |
| Estimated training time | 1-2 hours (GPU-only, no real env) |

**Actor-critic operates in latent space:**
The policy and value networks take the concatenated `(h, z)` vector (2560 dims) as input — NOT raw observations. This is much smaller and faster than our current CNN-based policy.

## Visualization & Demo

### Dream Viewer (primary demo)

Side-by-side window showing real game and world model reconstruction synchronized by action sequence:

```
┌──────────────────────┬──────────────────────┐
│    REAL GAME          │    MODEL'S DREAM      │
│    [pygame render     │    [decoded from       │
│     of actual game]   │     latent state]      │
│                       │                        │
│  Score: 3200          │  Predicted: 3180       │
│  Step: 847            │  Divergence: 0.023     │
└──────────────────────┴──────────────────────┘
```

Both sides receive identical actions. Divergence metric shows L2 distance between real and predicted states over time.

**Key visual moments to capture:**
- Ghost approaching — does the model predict movement correctly?
- Power pellet activation — do ghosts turn frightened in the dream?
- Death event — does the model predict termination?
- Long rollouts — where does the dream start diverging from reality?

### Training Dashboard Extension

Add world model metrics to the existing `dashboard.py`:
- Reconstruction loss curve
- KL divergence curve
- Reward prediction accuracy
- Continue prediction accuracy
- Dream agent score vs. real agent score (the headline metric)

### The Headline Metric

**"Dream accuracy: X%"** = dream_agent_score / real_agent_score * 100

This single number captures how well the world model has learned the game AND how well an agent can exploit that learned model. It's the number that goes in the tweet.

## File Structure

```
pacman/
├── world_model/                 ← NEW MODULE
│   ├── __init__.py
│   ├── rssm.py                  — RSSM: GRU dynamics + prior/posterior networks
│   ├── encoder.py               — CNN encoder: observation → latent
│   ├── decoder.py               — CNN decoder: latent → reconstructed observation
│   ├── heads.py                 — Reward and continue prediction MLPs
│   ├── world_model.py           — Integrates all components, train_step() and imagine()
│   └── replay_buffer.py         — Sequential episode storage, sample_sequences()
├── training/
│   ├── wm_trainer.py            ← NEW — world model training loop
│   └── dream_trainer.py         ← NEW — imagination PPO training loop
├── viz/
│   ├── dashboard.py             ← MODIFIED — add world model metrics tab
│   └── dream_viewer.py          ← NEW — side-by-side real vs. dream visualization
scripts/
├── collect_data.py              ← NEW — run trained agent, store episodes
├── train_world_model.py         ← NEW — train RSSM from collected data
├── train_dreamer.py             ← NEW — train agent in imagination
└── watch_dreams.py              ← NEW — launch dream viewer
```

**Estimated new code: ~1200-1500 lines**

| File | Est. lines | Complexity |
|------|-----------|------------|
| rssm.py | 150 | Core dynamics model |
| encoder.py | 80 | CNN encoder |
| decoder.py | 80 | Transposed CNN decoder |
| heads.py | 60 | Reward + continue MLPs |
| world_model.py | 200 | Integration, train_step, imagine |
| replay_buffer.py | 100 | Sequential episode storage |
| wm_trainer.py | 200 | World model training loop |
| dream_trainer.py | 150 | Imagination PPO loop |
| dream_viewer.py | 150 | Side-by-side pygame visualization |
| collect_data.py | 50 | Data collection script |
| train_world_model.py | 40 | CLI entry point |
| train_dreamer.py | 40 | CLI entry point |
| watch_dreams.py | 40 | CLI entry point |

## End-to-End Pipeline

```bash
# Step 1: Collect data from trained agent (~30 min, CPU)
python scripts/collect_data.py --checkpoint runs/.../best.pt --episodes 5000

# Step 2: Train world model (~8 hours, MPS GPU)
python scripts/train_world_model.py --data runs/.../replay_buffer/

# Step 3: Train dream agent (~2 hours, MPS GPU)
python scripts/train_dreamer.py --world-model runs/.../world_model.pt

# Step 4: Evaluate dream agent in real game
python scripts/evaluate.py --checkpoint runs/.../dream_agent.pt

# Step 5: Watch side-by-side comparison
python scripts/watch_dreams.py --real-agent runs/.../best.pt --dream-agent runs/.../dream_agent.pt
```

## Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Dream accuracy (score ratio) | >60% | >85% |
| Reconstruction loss | <0.05 MSE | <0.02 MSE |
| Reward prediction accuracy | >90% | >95% |
| Ghost position prediction (5-step) | >70% correct | >85% correct |
| Dream agent level clear rate | >50% | >80% |
| World model training time | <12 hours | <8 hours |

## Dependencies

No new dependencies required. Uses existing:
- PyTorch (MPS backend)
- NumPy
- PyYAML
- pygame (for visualization)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MPS backend numerical issues | Medium | High | Fall back to CPU for unstable ops; use float32 throughout |
| Reconstruction quality too low | Medium | Medium | Try symlog encoding, increase decoder capacity, add perceptual loss |
| Dream agent doesn't converge | Low | High | Tune imagination horizon, increase parallel dreams, add real-data mixing |
| KL collapse (posterior = prior trivially) | Medium | High | Free nats (1.0), KL balancing, monitor z usage |
| Memory pressure (24GB) | Low | Medium | Reduce batch size, use gradient checkpointing, chunk replay buffer |

## Future Extensions

After the core pipeline works:
- **Diffusion decoder** — replace transposed CNN with diffusion model for photorealistic dreams (DIAMOND-style)
- **Procedural mazes** — train world model on multiple layouts, test generalization
- **Self-play ghosts** — add learnable ghost policies, train via competitive self-play in imagination
- **Port to Crafter** — benchmark on public leaderboard for publishability
- **Blog post + Twitter thread** — with GIF comparisons of real vs. dreamed gameplay
