# Online Dream Loop — Iterative World Model + Dream Agent

## Goal

Close the dream-reality gap by iteratively collecting real data with the dream agent, fine-tuning the world model on that data, and retraining the dream agent. Each iteration makes the world model more accurate in the regions the dream agent actually visits, enabling the dream agent to improve further.

## Problem Statement

The current pipeline is **offline**: collect data once (PPO agent) → train world model once → train dream agent on frozen model. The dream agent peaks at ~536 real score at update 100, then degrades as it overfits to world model imperfections (score drops to 210 by update 300).

Root cause: the world model was trained on PPO agent data. The dream agent explores different state regions. The world model is inaccurate in those regions, and the dream agent exploits those inaccuracies.

## Architecture

### The Iterative Loop

```
For iteration i = 0 to N-1:
  1. Train dream agent on current world model
     - Fresh random policy each iteration
     - Early stop on real eval (patience=500, max 3000 updates)
     - Save best checkpoint as dream_agent_iter{i}_best.pt

  2. Collect real data with best dream agent
     - Deploy dream_agent_iter{i}_best.pt in real game
     - 300 episodes with stochastic sampling (temperature=1.0)
     - Store as single-frame obs (8ch), same format as PPO data

  3. Expand replay buffer
     - Append new episodes to existing buffer
     - Buffer grows: 1000 → 1300 → 1600 → ...

  4. Fine-tune world model
     - Load world model from previous iteration
     - 20K gradient steps on expanded buffer
     - LR=1e-4 (lower than initial 3e-4 to avoid forgetting)
     - Save as world_model_iter{i+1}.pt

  5. Check convergence
     - If iteration best score doesn't beat previous iteration, stop
```

Iteration 0 uses the existing world model (`world_model_latest.pt`) and replay buffer (`replay_buffer.pt`, 1000 PPO episodes).

### Dream Agent Data Collection

The dream agent operates differently from the PPO agent — it uses RSSM latent states, not frame-stacked observations. Collection flow per episode:

1. Reset real env, get single-frame observation
2. Encode obs with `wm.encoder` → initialize RSSM state `(h, z)`
3. At each step:
   - Form latent `[h; z]`, run `DreamPolicy` → sample action from logits
   - Step real env with action
   - Encode new obs → get posterior `z` from `wm.rssm.posterior(h, enc)`
   - Advance `h` with `wm.rssm.dynamics(h, z, action)`
4. Store single-frame obs, actions, rewards, dones

Output format is identical to `collect_data.py` — the replay buffer and world model don't know or care which agent generated the data.

### World Model Fine-tuning

Not retraining from scratch. The world model already understands game physics — it just needs to improve in the dream agent's operating regions.

- Load existing checkpoint (optimizer state included for momentum continuity)
- 20K gradient steps (vs 100K initial) — ~1.4 hours on MPS
- Learning rate: 1e-4 (vs 3e-4 initial)
- Same batch_size=16, seq_len=50
- Uniform sampling from replay buffer (no prioritization)
- Same loss function: recon + reward + continue + KL

### Dream Agent Retraining

Fresh random policy each iteration (not warm-started). Reasons:
- The world model changed — old value estimates are invalid
- Old policy may have learned to exploit quirks that are now fixed
- Fresh training with early stopping reaches peak quickly (~100-200 updates)

Per-iteration config:
- Horizon: 10 steps
- Imaginations: 512 parallel
- Entropy: 0.5 → 0.05 (annealed over 70% of training)
- Latent noise: σ=0.1
- Early stopping: patience=500 on real eval (eval every 50 updates)
- Max updates: 3000 per iteration
- LR: 3e-5

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `scripts/train_online.py` | Orchestrator: runs the iterative loop |
| `scripts/collect_dream_data.py` | Collects gameplay data using the dream agent |

### Modified Files

| File | Change |
|------|--------|
| `pacman/training/wm_trainer.py` | Add `fine_tune()` method that accepts an existing checkpoint and uses lower LR |
| `pacman/training/dream_trainer.py` | `train()` returns a dict `{"best_score": float, "best_update": int, "best_path": Path}` |

### Unchanged Files

All world model components (RSSM, encoder, decoder, heads, replay buffer), PPO agent, game engine, and environments remain untouched.

## Expected Results

| Iteration | WM Data | WM Steps | Dream Agent Best Score |
|-----------|---------|----------|----------------------|
| 0 | 1000 ep (PPO) | 100K (existing) | 536 (existing) |
| 1 | +300 ep (dream) | 20K fine-tune | ~700-900 |
| 2 | +300 ep (dream) | 20K fine-tune | ~900-1200 |
| 3 | +300 ep (dream) | 20K fine-tune | ~1200+ or plateau |

These are estimates. The key metric is whether each iteration's best score exceeds the previous. If it doesn't, the loop stops automatically.

## Compute Budget

Per iteration (after iteration 0):
- Data collection: ~15 min (300 episodes)
- WM fine-tune: ~1.4 hours (20K steps at 4 steps/s)
- Dream training: ~15 min (3000 updates max, early stop likely at ~500)
- **Total: ~2 hours per iteration**

For 3 iterations: ~6 hours total. For 5 iterations: ~10 hours.

## Success Criteria

1. Dream agent real eval score improves across iterations (primary metric)
2. World model recon/reward losses decrease or remain stable during fine-tuning (no catastrophic forgetting)
3. Early stopping triggers later in each iteration (the dream-reality gap is smaller, so the agent can train longer before overfitting)

## Convergence / Stopping

The loop stops when either:
- An iteration's best score doesn't beat the previous iteration's best
- Maximum iterations reached (default: 5)
- User interrupts (Ctrl+C — saves current best checkpoints)
