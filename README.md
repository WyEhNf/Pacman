# Pac-Man Deep RL Experimental Analysis

A systematic experimental comparison of three deep reinforcement learning paradigms—DecisionTransformer (offline), DAgger+DQN (imitation), and PPO+GAE (on-policy)—on the classic arcade game Pac-Man.

## Project Structure

```
Pacman/
├── agents/
│   ├── ppo/                  # PPO+GAE training pipeline (arcade-accurate simulator)
│   │   ├── pacman/           # Core engine: game, env, agents, training
│   │   │   ├── engine/       # 28×31 arcade simulator (ghost AI, maze, game logic)
│   │   │   ├── env/          # Single & vectorized environments with frame stacking
│   │   │   ├── agents/       # Actor-Critic, PPO, RND, rollout buffer
│   │   │   └── training/     # Trainer, evaluator, checkpoint utils
│   │   ├── scripts/          # Entry points: train.py, evaluate.py, dashboard.py
│   │   ├── configs/          # YAML training configurations
│   │   └── runs/             # Training run outputs (checkpoints, tensorboard)
│   │
│   ├── dt/                   # DecisionTransformer implementation
│   │   └── src/model/        # GPT-2-style Transformer, BC pre-training, TD fine-tuning
│   │
│   ├── DAgger/               # DAgger+DQN pipeline
│   │   ├── collect_*.py      # Expert data collection scripts (multiple strategies)
│   │   └── eval_*.py         # Ensemble evaluation & model comparison
│   │
│   ├── checkpoints/          # Saved model weights for DT, DQN, and DAgger
│   └── experts_data/         # Expert trajectory datasets (.npz, gitignored)
│
├── Berkeley's/               # Berkeley Pac-Man framework (PPCA-AIPacMan-2024)
│   ├── reinforcement/        # Q-learning, approximate Q-learning, feature extractors
│   ├── search/               # A*, minimax, alpha-beta search agents
│   ├── multiagent/           # Multi-agent Pac-Man (ghosts as independent agents)
│   └── logic/                # First-order logic planning
│
├── related work/
│   ├── Microsoft_HRA/        # van Seijen et al. "Hybrid Reward Architecture" (NeurIPS 2017)
│   └── Stanford-CS221-pacman_with_hra/  # CS221 course project with HRA integration
│
├── configs/                  # Shared configuration files
├── loggers/                  # Experiment logs, verification scripts, and Phase 2 plan
├── Experimental Analysis.tex # Technical report (LaTeX)
└── Experimental Analysis.pdf # Compiled technical report
```

## Architectures Compared

### DecisionTransformer (Offline → Online)
- GPT-2-style causal Transformer conditioned on (return-to-go, state, action)
- Stage 1: Behavioral cloning on expert trajectories
- Stage 2: Online fine-tuning with TD-error
- **Platform**: Berkeley Pac-Man engine (`smallGrid`, agent-centered feature vectors)

### DAgger + DQN (Imitation + Correction)
- Pre-trained PPO policy serves as teacher; DQN student collects online rollouts
- Teacher annotates student-visited states → mixed dataset → Bellman updates
- **Platform**: Berkeley Pac-Man engine (`smallGrid` / `mediumClassic`)

### PPO + GAE (On-Policy Self-Play)
- CNN backbone (3-layer Conv2d), 128 parallel environments
- GAE credit assignment ($\gamma=0.99$, $\lambda=0.95$), RND curiosity-driven exploration
- Curriculum learning: Difficulty 0 → 1 → 2 (Scatter-only → Scatter/Chase → Cruise Elroy)
- **Platform**: Custom arcade-accurate $28\times 31$ simulator with 4 deterministic ghost AIs

### Enhanced PPO (Phase 3 — In Training)
- 18-channel observation space with explicit spatial priors:
  - Per-ghost pursuit targets (Ch4–7) and one-step forward predictions (Ch14–17)
  - Eaten-ghost visibility (Ch9), spatial direction arrow (Ch13)
- 15-dimensional scalar features including per-ghost distances and directions
- Multi-objective reward function: death penalty $\times 8$ ($-80$), exponential ghost-hunting incentives ($20\!\to\!40\!\to\!80\!\to\!160$), edible-ghost proximity gradients

## Key Findings

| Architecture | Platform | 3-Life Clear | Characteristic Failure Mode |
|:---|:---|:---:|:---|
| DecisionTransformer | Berkeley smallGrid | 5% | Offline–online coverage gap under skewed returns |
| DAgger+DQN | Berkeley mediumClassic | 20%* | Teacher-ceiling collapse when layout difficulty exceeds teacher competence |
| PPO+GAE (baseline) | Arcade 28×31 | **95%** | Representational bottleneck (0% 1-life clearance) |

*\*DAgger on mediumClassic (simpler layout): 389 → 625 pts. On harder layouts, Q-value 3→214 while score 278→−356.*

**Cross-cutting insight**: Offline and imitation-based methods fail because their learning signal (CE loss) decouples from the consequences of the policy's own actions. On-policy TD-based methods self-correct: bad states receive negative advantages, and the policy adjusts in the next rollout.

## Setup

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- NumPy, PyYAML, TensorBoard

### Installation
```bash
# For PPO training
cd agents/ppo
pip install -e .

# For DT and DAgger (Berkeley engine)
cd agents/dt
pip install -e .
```

### Training PPO (Arcade Simulator)
```bash
cd agents/ppo

# From scratch (8,000 updates, ~37 hours on RTX 5070)
python scripts/train.py --config configs/phase3_unified.yaml --run-dir runs/my_run

# Resume from checkpoint
python scripts/train.py --config configs/phase3_unified.yaml --run-dir runs/my_run --resume

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/latest.pt --episodes 100
```

### Evaluating DAgger / DT (Berkeley Engine)
```bash
cd agents/DAgger
python eval_cnn_ensemble.py           # CNN ensemble evaluation
python eval_compare_three.py          # Cross-model comparison
```

## Results & Logs

Training metrics are logged to TensorBoard under each run directory:
```
agents/ppo/runs/<run_name>/tensorboard/
```

Key experiment logs (in `loggers/`):
- `td_finetune_log.txt` — DecisionTransformer fine-tuning
- `phaseB_v3_log.txt`, `phaseB_v4_log.txt` — DAgger+DQN degradation
- `phaseB_rwr_log.txt` — DAgger on mediumClassic (positive control)
- `selfplay_log.txt` — Self-play DAgger experiments
- `hra_paper_log.txt` — HRA reward decomposition

## Reference Implementations

- `related work/Microsoft_HRA/` — van Seijen et al. (NeurIPS 2017)
- `related work/Stanford-CS221-pacman_with_hra/` — CS221 HRA integration

More noted in the report.

