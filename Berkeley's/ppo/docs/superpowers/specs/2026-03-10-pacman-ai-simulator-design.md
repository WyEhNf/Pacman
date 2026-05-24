# Pac-Man AI Simulator — Design Specification

## Overview

Research-grade Pac-Man AI simulator where 5 independent DQN agents (1 Pac-Man + 4 ghosts) learn through continuous self-play. Optimized for training throughput on M4 MacBook Air with MPS acceleration. GUI is secondary — for spot-checking live gameplay. Primary goal: observe and measure whether Pac-Man or the ghost team becomes superior over thousands of games.

## Architecture

```
CONFIG (YAML)
    │
TRAINING LOOP (orchestrator)
    ├── GAME ENGINE (pure Python, tick-based, Gym-like step() API)
    ├── OBSERVATION BUILDER (game state → per-agent feature vectors)
    ├── 5 × DQN AGENTS (independent networks, replay buffers, optimizers)
    ├── DATA LOGGER (SQLite metrics, JSONL events)
    └── GUI (optional Pygame game view + web metrics dashboard)
```

**Dependency graph (strict):**
- `engine/` → imports nothing from other modules. Pure game logic.
- `agents/` → imports engine constants/types only. PyTorch dependency.
- `training/` → imports engine/ and agents/. Calls data/ for logging.
- `gui/` → imports engine/ for rendering state. Never imported by training/.
- `data/` → standalone. Imported by training/.

**Three modes:**
- `train` — headless, max speed, epsilon-greedy exploration, learning enabled
- `eval` — headless, no exploration, metrics only, no learning
- `watch` — Pygame GUI attached, slowed to human speed, can toggle learning

## Game Engine

### Maze
- Classic 28×31 grid
- Walls, pellets (240), power pellets (4), ghost house, tunnels (2 wrap-around)
- Maze stored as 2D integer grid in `maze_data.py`
- Tile types: WALL, EMPTY, PELLET, POWER_PELLET, GHOST_HOUSE, GHOST_DOOR, TUNNEL

### Entities
- **Pac-Man:** position, direction, lives (3), score, powered-up timer
- **Ghosts (4):** Blinky, Pinky, Inky, Clyde — each with position, direction, mode, scatter target corner, home position in ghost house
- **Fruit:** spawns at fixed position after 70 and 170 pellets eaten, disappears after timer

### Ghost Modes
- **Scatter:** ghosts move toward their assigned corner (classic behavior timer-based)
- **Chase:** ghosts pursue Pac-Man (default active mode)
- **Frightened:** triggered by power pellet, all ghosts become vulnerable, timer-based duration
- **Eaten:** ghost returns to ghost house after being eaten, then respawns

### Mode Timer Schedule (classic)
Scatter/Chase alternation per level:
- Scatter 7s → Chase 20s → Scatter 7s → Chase 20s → Scatter 5s → Chase 20s → Scatter 5s → Chase ∞
- Frightened duration: 6 seconds (configurable)

### Movement
- Tick-based discrete grid movement
- Ghosts cannot reverse direction (except when mode changes)
- Tunnel wrapping at maze edges
- Ghost house: ghosts exit sequentially based on pellet-count triggers (Blinky starts outside)

### Collision Detection
- Pac-Man + Pellet → pellet consumed, score +10
- Pac-Man + Power Pellet → consumed, score +50, all ghosts enter frightened mode
- Pac-Man + Ghost (normal) → Pac-Man loses a life, positions reset
- Pac-Man + Ghost (frightened) → ghost eaten, score +200/400/800/1600 (escalating), ghost enters eaten mode
- Pac-Man + Fruit → fruit consumed, bonus score

### Win/Lose
- **Win (Pac-Man):** all pellets eaten
- **Lose (Pac-Man):** all lives lost (ghosts win)
- After either: automatic reset and new episode begins

## Agents

### DQN Architecture (per agent)
```
Input (~40-50 features) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(4)
Output: Q-values for [UP, DOWN, LEFT, RIGHT]
~55K parameters per agent, ~275K total
```

### DQN Components (per agent)
- **Q-Network** — online network for action selection
- **Target Network** — soft-updated copy (τ=0.005) for stable targets
- **Replay Buffer** — 100K transitions, uniform sampling (prioritized replay deferred to v2)
- **Optimizer** — Adam, lr=3e-4
- **Loss** — Huber (smooth L1)
- **Gradient clipping** — max norm 1.0
- **Exploration** — ε-greedy, linear decay 1.0→0.05 over 500 episodes

### Action Masking
Invalid actions (moving into walls) are masked by setting Q-values to -∞ before argmax. Agents only choose legal moves.

### Pac-Man Observation Vector (~50 features)
| Feature | Size | Description |
|---------|------|-------------|
| Position (x, y) | 2 | Normalized grid position |
| Wall sensors | 4 | Legal moves in 4 directions |
| Per-ghost info ×4 | 16 | Relative dx, dy, distance, mode one-hot (chase/scatter/frightened/eaten) |
| Nearest pellet | 4 | Direction + distance |
| Nearest power pellet | 4 | Direction + distance |
| Pellet density (4 quadrants) | 4 | Pellet count per quadrant relative to Pac-Man |
| Game progress | 2 | Pellets remaining fraction, lives normalized |
| Fruit info | 3 | Active flag, relative direction |

**Critical:** Ghost mode is a raw state feature. Pac-Man is NOT told what modes mean. It must learn through reward that "frightened" ghosts yield +5 reward instead of -10 death.

### Ghost Observation Vector (~40 features per ghost)
| Feature | Size | Description |
|---------|------|-------------|
| Own position | 2 | Normalized grid position |
| Own mode | 4 | One-hot: chase/scatter/frightened/eaten |
| Wall sensors | 4 | Legal moves |
| Pac-Man info | 4 | Relative dx, dy, distance, powered-up flag |
| Other ghosts ×3 | 12 | Relative dx, dy, distance, mode per teammate |
| Scatter target | 2 | Direction to scatter corner |
| Game state | 2 | Pellets remaining, frightened timer |

### Reward Design

**Pac-Man:**
| Event | Reward |
|-------|--------|
| Eat pellet | +1.0 |
| Eat power pellet | +2.0 |
| Eat ghost | +5.0 |
| Eat fruit | +3.0 |
| Clear level | +20.0 |
| Caught by ghost | -10.0 |
| Game over | -20.0 |
| Time step | -0.01 |

**Ghost (per ghost):**
| Event | Reward |
|-------|--------|
| Catch Pac-Man (self) | +10.0 |
| Team catches Pac-Man | +3.0 |
| Game over (ghosts win) | +15.0 |
| Pac-Man eats pellet | -0.5 |
| Pac-Man clears level | -15.0 |
| Got eaten (frightened) | -5.0 |
| Proximity shaping | +0.1 × (1/distance) |

All reward values configurable via YAML.

## Training

### Hyperparameters (all configurable)
| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Batch size | 64 |
| Discount (γ) | 0.99 |
| ε start→end | 1.0→0.05 |
| ε decay episodes | 500 |
| Replay buffer | 100,000 |
| Target update τ | 0.005 |
| Min replay before learn | 1,000 |
| Learn every N steps | 4 |
| Checkpoint every | 100 episodes |
| Max steps/episode | 3,000 |
| Lives | 3 |

### Training Loop
1. Reset game environment
2. While episode not done and steps < max:
   a. Each agent observes its state
   b. Each agent selects action (ε-greedy)
   c. Engine steps with all actions
   d. Each agent receives reward and next state
   e. Each agent stores transition in replay buffer
   f. Every 4 steps: each agent samples batch and learns
3. Log episode metrics
4. Checkpoint every 100 episodes
5. Start next episode

### Checkpointing
Per agent: model weights, target weights, optimizer state, epsilon value, replay buffer (optional — can be large).
Global: episode count, training config snapshot, random seeds.

## Data Collection

### SQLite Database (`metrics.db`)
**Tables:**
- `episodes` — episode_id, timestamp, winner, score, steps, pellets_eaten, ghosts_eaten, fruits_eaten, lives_remaining, level_cleared
- `agent_metrics` — episode_id, agent_name, total_reward, avg_q_value, epsilon, loss, actions_taken
- `training_config` — key-value config snapshot

### Event Log (`events.jsonl`)
Optional per-step event log for detailed analysis: step, agent, action, reward, position, game_event.

## GUI

### Pygame Game Renderer
- 28×31 grid rendered with colored tiles
- Pac-Man: yellow circle with mouth animation
- Ghosts: colored shapes (Blinky=red, Pinky=pink, Inky=cyan, Clyde=orange)
- Frightened ghosts: blue color + visual blinking near timer end
- Eaten ghosts: eyes only
- Pellets: small dots, power pellets: large flashing dots
- Fruit: colored symbol
- Score, lives, episode number displayed

### Web Metrics Dashboard (Dash/Plotly)
- Win rate over time (Pac-Man vs ghosts, rolling window)
- Per-agent reward curves
- Per-ghost performance comparison
- Score trends
- Survival time, pellets cleared, ghost captures
- Epsilon decay visualization
- Auto-refreshing from SQLite

## Project Structure

```
pacman-ai/
├── config/default.yaml
├── src/
│   ├── engine/    (maze.py, entities.py, game.py, constants.py, maze_data.py)
│   ├── agents/    (base_agent.py, dqn_agent.py, replay_buffer.py, networks.py, observations.py)
│   ├── training/  (trainer.py, evaluator.py, checkpoint.py)
│   ├── gui/       (renderer.py, sprites.py, dashboard.py)
│   ├── data/      (logger.py, schemas.py, analyzer.py)
│   └── utils/     (config.py, seeding.py)
├── tests/         (test_maze.py, test_game.py, test_agents.py, test_observations.py, etc.)
├── scripts/       (train.py, evaluate.py, watch.py, dashboard.py)
├── runs/          (gitignored, per-run output directories)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Tech Stack
- **Python 3.11+**
- **PyTorch** — DQN networks, MPS backend for M4 GPU
- **Pygame** — game rendering
- **Dash + Plotly** — web metrics dashboard
- **SQLite** — metrics storage (stdlib, no extra dependency)
- **PyYAML** — config loading
- **NumPy** — array ops, maze representation
- **pytest** — testing

## Performance Targets
- ~2-5 games/second headless on M4
- ~1,000 games in 5-10 minutes
- ~10,000 games in under 2 hours
- SQLite DB stays under 100MB for 10K games
- Total replay buffers ~2GB RAM for 5 agents at 100K

## v2 Improvements (deferred)
- Double DQN (reduce Q-value overestimation)
- Prioritized experience replay
- Dueling network architecture
- Multiple maze layouts
- Curriculum learning (progressive difficulty)
- Centralized training with decentralized execution (CTDE) for ghosts
- Tensorboard integration
- Hyperparameter sweep tooling
