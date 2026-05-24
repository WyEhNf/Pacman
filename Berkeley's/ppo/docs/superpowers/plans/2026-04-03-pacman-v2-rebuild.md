# Pac-Man v2 Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Pac-Man AI simulator from scratch with PPO + CNN observations + vectorized NumPy engine for maximum AI performance.

**Architecture:** Single PPO agent (Pac-Man) trained against classic scripted ghost AI. 8-channel CNN observations on the raw 31x28 grid. Vectorized engine steps 64 games simultaneously as batched NumPy operations. Actor-critic network with shared CNN backbone (~1.65M params).

**Tech Stack:** Python 3.10+, PyTorch (MPS), NumPy, Pygame, TensorBoard, Dash/Plotly, PyYAML

**Spec:** `docs/superpowers/specs/2026-04-03-pacman-v2-rebuild-design.md`

---

## File Structure

New `pacman/` package (v1 code in `src/` left untouched):

```
pacman/
  __init__.py
  engine/
    __init__.py
    constants.py         # Task 1
    maze_data.py         # Task 2 (carried from v1)
    maze.py              # Task 2
    entities.py          # Task 3
    ghost_ai.py          # Task 4
    game.py              # Task 5
  env/
    __init__.py
    pacman_env.py        # Task 7
    vec_env.py           # Task 8
  agents/
    __init__.py
    networks.py          # Task 9
    rollout.py           # Task 10
    ppo.py               # Task 11
  training/
    __init__.py
    checkpoint.py        # Task 12
    evaluator.py         # Task 13
    trainer.py           # Task 14
  viz/
    __init__.py
    sprites.py           # Task 15
    renderer.py          # Task 16
    dashboard.py         # Task 17
  utils/
    __init__.py
    config.py            # Task 0
config/
  default.yaml           # Task 0
scripts/
  train.py               # Task 18
  evaluate.py            # Task 18
  watch.py               # Task 18
tests/
  test_engine.py         # Task 6
  test_vec_env.py        # Task 8
  test_networks.py       # Task 9
  test_ppo.py            # Task 11
  test_training.py       # Task 14
pyproject.toml           # Task 0
requirements.txt         # Task 0
```

---

### Task 0: Project Setup — Config, Dependencies, Package Scaffolding

**Files:**
- Create: `pacman/__init__.py`, `pacman/engine/__init__.py`, `pacman/env/__init__.py`, `pacman/agents/__init__.py`, `pacman/training/__init__.py`, `pacman/viz/__init__.py`, `pacman/utils/__init__.py`, `pacman/utils/config.py`
- Modify: `pyproject.toml`, `requirements.txt`
- Create: `pacman/config/default.yaml`

- [ ] **Step 1: Create package directories and `__init__.py` files**

Create every `__init__.py` as empty files:

```python
# pacman/__init__.py — empty
# pacman/engine/__init__.py — empty
# pacman/env/__init__.py — empty
# pacman/agents/__init__.py — empty
# pacman/training/__init__.py — empty
# pacman/viz/__init__.py — empty
# pacman/utils/__init__.py — empty
```

- [ ] **Step 2: Create config loader**

```python
# pacman/utils/config.py
from pathlib import Path
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 3: Create `pacman/config/default.yaml`**

```yaml
game:
  lives: 3
  max_steps: 3000
  frightened_duration: 36
  mode_schedule: [42, 120, 42, 120, 30, 120, 30, -1]
  ghost_exit_pellets: [0, 30, 60, 80]
  fruit:
    spawn_pellets: [70, 170]
    duration: 60
    score: 100

env:
  num_envs: 64
  observation_channels: 8
  num_scalar_features: 4

network:
  cnn_channels: [32, 64, 64]
  cnn_kernels: [3, 3, 3]
  cnn_strides: [1, 2, 2]
  shared_hidden: 512
  head_hidden: 128

ppo:
  rollout_steps: 128
  num_epochs: 4
  minibatch_size: 512
  learning_rate: 2.5e-4
  lr_anneal: true
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_loss_coef: 0.5
  entropy_coef_start: 0.01
  entropy_coef_end: 0.001
  entropy_anneal_fraction: 0.6
  max_grad_norm: 0.5
  reward_clip: 10.0

curriculum:
  phase_thresholds: [0, 500, 2000]
  difficulties: [0, 1, 2]

rewards:
  eat_pellet: 1.0
  eat_power_pellet: 2.0
  eat_ghost: [5.0, 10.0, 15.0, 20.0]
  eat_fruit: 3.0
  clear_level: 50.0
  death: -10.0
  game_over: -25.0
  time_step: -0.01

training:
  total_updates: 5000
  eval_every: 50
  eval_episodes: 20
  checkpoint_every: 100
  log_every: 1

device: auto
```

- [ ] **Step 4: Update `pyproject.toml`**

Add `pacman` to package discovery. Update dependencies to include `tensorboard`.

- [ ] **Step 5: Update `requirements.txt`**

```
torch>=2.0
numpy>=1.24
pygame>=2.5
pyyaml>=6.0
tensorboard>=2.14
dash>=2.14
plotly>=5.18
```

- [ ] **Step 6: Install and verify**

Run: `pip install -e .`
Expected: installs successfully. `python -c "from pacman.utils.config import load_config; c = load_config(); print(c['ppo']['learning_rate'])"` prints `0.00025`.

- [ ] **Step 7: Commit**

```bash
git add pacman/ config/ pyproject.toml requirements.txt
git commit -m "feat(v2): scaffold pacman package, config, dependencies"
```

---

### Task 1: Engine Constants — `pacman/engine/constants.py`

**Files:**
- Create: `pacman/engine/constants.py`

- [ ] **Step 1: Write constants module**

Carry forward v1 enums with additions for vectorized ops and arcade ghost AI:

```python
# pacman/engine/constants.py
from enum import IntEnum
import numpy as np


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Tile(IntEnum):
    WALL = 0
    EMPTY = 1
    PELLET = 2
    POWER_PELLET = 3
    GHOST_HOUSE = 4
    GHOST_DOOR = 5
    TUNNEL = 6


class GhostMode(IntEnum):
    SCATTER = 0
    CHASE = 1
    FRIGHTENED = 2
    EATEN = 3


class GhostID(IntEnum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    CLYDE = 3


# Direction deltas: (row_delta, col_delta) — NumPy array for vectorized ops
DIRECTION_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16)

# Dict form for single-game use
DIRECTION_DELTA_MAP = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

OPPOSITE_DIRECTION = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Arcade ghost tie-breaking priority: UP > LEFT > DOWN > RIGHT
DIRECTION_PRIORITY = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]

# Ghost scatter target corners (row, col)
SCATTER_TARGETS = {
    GhostID.BLINKY: (0, 25),     # top-right
    GhostID.PINKY: (0, 2),       # top-left
    GhostID.INKY: (30, 27),      # bottom-right
    GhostID.CLYDE: (30, 0),      # bottom-left
}

# Ghost colors for rendering
GHOST_COLORS = {
    GhostID.BLINKY: (255, 0, 0),
    GhostID.PINKY: (255, 184, 255),
    GhostID.INKY: (0, 255, 255),
    GhostID.CLYDE: (255, 184, 82),
}
FRIGHTENED_COLOR = (33, 33, 222)
FRIGHTENED_FLASH_COLOR = (255, 255, 255)

# Maze dimensions
MAZE_ROWS = 31
MAZE_COLS = 28
NUM_ACTIONS = 4
NUM_GHOSTS = 4

# Cruise Elroy thresholds (pellets remaining)
ELROY_1_THRESHOLD = 20
ELROY_2_THRESHOLD = 10
```

- [ ] **Step 2: Verify imports**

Run: `python -c "from pacman.engine.constants import Direction, DIRECTION_DELTAS; print(DIRECTION_DELTAS.shape)"`
Expected: `(4, 2)`

- [ ] **Step 3: Commit**

```bash
git add pacman/engine/constants.py
git commit -m "feat(v2): engine constants — enums, deltas, ghost config"
```

---

### Task 2: Maze Data + Maze Utilities — `pacman/engine/maze_data.py`, `pacman/engine/maze.py`

**Files:**
- Create: `pacman/engine/maze_data.py` (carry from v1 `src/engine/maze_data.py`)
- Create: `pacman/engine/maze.py`

- [ ] **Step 1: Copy maze data from v1**

Copy `src/engine/maze_data.py` to `pacman/engine/maze_data.py`. This contains `CLASSIC_MAZE`, `PACMAN_START`, `GHOST_START_POSITIONS`, `GHOST_HOME_POSITIONS`, `GHOST_DOOR_POS`, `FRUIT_POSITION`, `TUNNEL_POSITIONS`. No modifications needed — it's pure data.

- [ ] **Step 2: Write maze utility functions**

```python
# pacman/engine/maze.py
import numpy as np
from collections import deque

from .constants import Tile, Direction, DIRECTION_DELTAS, MAZE_ROWS, MAZE_COLS
from .maze_data import CLASSIC_MAZE, GHOST_DOOR_POS


def load_initial_grid() -> np.ndarray:
    """Return a fresh (31, 28) int8 grid from the classic maze layout."""
    return np.array(CLASSIC_MAZE, dtype=np.int8)


def count_pellets(grid: np.ndarray) -> int:
    """Count total pellets (regular + power) in a grid."""
    return int(np.sum((grid == Tile.PELLET) | (grid == Tile.POWER_PELLET)))


def count_power_pellets(grid: np.ndarray) -> int:
    """Count power pellets in a grid."""
    return int(np.sum(grid == Tile.POWER_PELLET))


def is_walkable(grid: np.ndarray, row: int, col: int, for_ghost: bool = False) -> bool:
    """Check if a tile is walkable."""
    if row < 0 or row >= MAZE_ROWS or col < 0 or col >= MAZE_COLS:
        return False
    tile = grid[row, col]
    if tile == Tile.WALL:
        return False
    if not for_ghost and tile in (Tile.GHOST_HOUSE, Tile.GHOST_DOOR):
        return False
    return True


def get_legal_directions(grid: np.ndarray, row: int, col: int,
                         for_ghost: bool = False) -> list[Direction]:
    """Return list of legal movement directions from a position."""
    legal = []
    for d in Direction:
        dr, dc = DIRECTION_DELTAS[d]
        nr, nc = row + dr, col + dc
        # Tunnel wrapping
        nc = nc % MAZE_COLS
        if is_walkable(grid, nr, nc, for_ghost):
            legal.append(d)
    return legal


def compute_ghost_return_paths(grid: np.ndarray) -> np.ndarray:
    """BFS from ghost door to compute O(1) return direction lookup.

    Returns (31, 28) int8 array where each cell contains the Direction
    to move to get closer to the ghost door. -1 for unreachable/wall cells.
    """
    door_row, door_col = GHOST_DOOR_POS
    return_dirs = np.full((MAZE_ROWS, MAZE_COLS), -1, dtype=np.int8)
    return_dirs[door_row, door_col] = 0  # at destination

    visited = np.zeros((MAZE_ROWS, MAZE_COLS), dtype=bool)
    visited[door_row, door_col] = True
    queue = deque([(door_row, door_col)])

    while queue:
        r, c = queue.popleft()
        for d in Direction:
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = r + dr, c + dc
            nc = nc % MAZE_COLS
            if 0 <= nr < MAZE_ROWS and not visited[nr, nc]:
                tile = grid[nr, nc]
                if tile != Tile.WALL:
                    visited[nr, nc] = True
                    # The direction to return FROM (nr, nc) is the OPPOSITE
                    # of the direction we expanded (toward the door)
                    opp = d ^ 1  # UP<->DOWN (0<->1), LEFT<->RIGHT (2<->3)
                    return_dirs[nr, nc] = opp
                    queue.append((nr, nc))

    return return_dirs
```

- [ ] **Step 3: Verify**

Run: `python -c "from pacman.engine.maze import load_initial_grid, count_pellets, compute_ghost_return_paths; g = load_initial_grid(); print(g.shape, count_pellets(g)); rp = compute_ghost_return_paths(g); print(rp.shape)"`
Expected: `(31, 28) 244` (or similar pellet count), `(31, 28)`

- [ ] **Step 4: Commit**

```bash
git add pacman/engine/maze_data.py pacman/engine/maze.py
git commit -m "feat(v2): maze data + utilities — grid loader, BFS return paths"
```

---

### Task 3: Entities — `pacman/engine/entities.py`

**Files:**
- Create: `pacman/engine/entities.py`

- [ ] **Step 1: Write GameState dataclass**

```python
# pacman/engine/entities.py
from dataclasses import dataclass, field
import numpy as np

from .constants import (
    Direction, GhostMode, GhostID, NUM_GHOSTS, MAZE_ROWS, MAZE_COLS,
)
from .maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, GHOST_HOME_POSITIONS,
    GHOST_DOOR_POS, FRUIT_POSITION,
)
from .maze import load_initial_grid, count_pellets


@dataclass
class GameState:
    """All mutable state for a single Pac-Man game."""
    grid: np.ndarray                    # (31, 28) int8
    pac_pos: np.ndarray                 # (2,) int16 — [row, col]
    pac_dir: int                        # Direction enum value
    pac_lives: int
    pac_powered: bool
    pac_power_timer: int
    pac_ghosts_eaten: int               # ghosts eaten during current power-up
    ghost_pos: np.ndarray               # (4, 2) int16
    ghost_dir: np.ndarray               # (4,) int8
    ghost_mode: np.ndarray              # (4,) int8
    ghost_fright_timer: np.ndarray      # (4,) int16
    ghost_in_house: np.ndarray          # (4,) bool
    ghost_exiting: np.ndarray           # (4,) bool
    score: int
    step_count: int
    pellets_eaten: int
    total_pellets: int
    pellets_remaining: int
    done: bool
    winner: str | None                  # "pacman", "ghosts", or None
    # Mode schedule state
    mode_index: int
    mode_timer: int
    global_mode: int                    # GhostMode.SCATTER or CHASE
    # Fruit state
    fruit_active: bool
    fruit_timer: int
    fruit_spawned: list                 # which thresholds have triggered
    # Difficulty (curriculum)
    difficulty: int


def create_initial_state(config: dict, difficulty: int = 0) -> GameState:
    """Create a fresh game state from config."""
    grid = load_initial_grid()
    total = count_pellets(grid)

    ghost_pos = np.array(
        [GHOST_START_POSITIONS[GhostID(i)] for i in range(NUM_GHOSTS)],
        dtype=np.int16,
    )
    ghost_dir = np.full(NUM_GHOSTS, Direction.LEFT, dtype=np.int8)
    ghost_mode = np.full(NUM_GHOSTS, GhostMode.SCATTER, dtype=np.int8)
    ghost_fright_timer = np.zeros(NUM_GHOSTS, dtype=np.int16)
    # Blinky starts outside, others inside ghost house
    ghost_in_house = np.array(
        [i != GhostID.BLINKY for i in range(NUM_GHOSTS)], dtype=bool,
    )
    ghost_exiting = np.zeros(NUM_GHOSTS, dtype=bool)

    return GameState(
        grid=grid,
        pac_pos=np.array(PACMAN_START, dtype=np.int16),
        pac_dir=Direction.LEFT,
        pac_lives=config["game"]["lives"],
        pac_powered=False,
        pac_power_timer=0,
        pac_ghosts_eaten=0,
        ghost_pos=ghost_pos,
        ghost_dir=ghost_dir,
        ghost_mode=ghost_mode,
        ghost_fright_timer=ghost_fright_timer,
        ghost_in_house=ghost_in_house,
        ghost_exiting=ghost_exiting,
        score=0,
        step_count=0,
        pellets_eaten=0,
        total_pellets=total,
        pellets_remaining=total,
        done=False,
        winner=None,
        mode_index=0,
        mode_timer=config["game"]["mode_schedule"][0],
        global_mode=GhostMode.SCATTER,
        fruit_active=False,
        fruit_timer=0,
        fruit_spawned=[],
        difficulty=difficulty,
    )


def reset_positions(state: GameState, config: dict) -> None:
    """Reset entity positions after Pac-Man death (in-place)."""
    state.pac_pos[:] = PACMAN_START
    state.pac_dir = Direction.LEFT
    state.pac_powered = False
    state.pac_power_timer = 0
    state.pac_ghosts_eaten = 0
    for i in range(NUM_GHOSTS):
        state.ghost_pos[i] = GHOST_START_POSITIONS[GhostID(i)]
        state.ghost_dir[i] = Direction.LEFT
        state.ghost_mode[i] = state.global_mode
        state.ghost_fright_timer[i] = 0
        state.ghost_in_house[i] = (i != GhostID.BLINKY)
        state.ghost_exiting[i] = False
```

- [ ] **Step 2: Verify**

Run: `python -c "from pacman.utils.config import load_config; from pacman.engine.entities import create_initial_state; s = create_initial_state(load_config()); print(s.pac_pos, s.ghost_pos.shape, s.total_pellets)"`

- [ ] **Step 3: Commit**

```bash
git add pacman/engine/entities.py
git commit -m "feat(v2): GameState dataclass + factory functions"
```

---

### Task 4: Ghost AI — `pacman/engine/ghost_ai.py`

**Files:**
- Create: `pacman/engine/ghost_ai.py`

- [ ] **Step 1: Write ghost target + movement functions**

```python
# pacman/engine/ghost_ai.py
"""Classic arcade ghost AI — faithful to the 1980 Pac-Man behaviors."""
import numpy as np

from .constants import (
    Direction, GhostMode, GhostID, DIRECTION_DELTAS, DIRECTION_PRIORITY,
    OPPOSITE_DIRECTION, SCATTER_TARGETS, MAZE_COLS, NUM_GHOSTS,
    ELROY_1_THRESHOLD, ELROY_2_THRESHOLD,
)
from .maze import is_walkable


def compute_chase_target(
    ghost_id: int,
    pac_pos: np.ndarray,
    pac_dir: int,
    ghost_pos: np.ndarray,   # (4, 2) all ghost positions
) -> tuple[int, int]:
    """Compute chase target tile for a specific ghost."""
    pr, pc = int(pac_pos[0]), int(pac_pos[1])

    if ghost_id == GhostID.BLINKY:
        # Direct pursuit — target Pac-Man's exact tile
        return pr, pc

    elif ghost_id == GhostID.PINKY:
        # Ambush — 4 tiles ahead of Pac-Man's facing direction
        dr, dc = DIRECTION_DELTAS[pac_dir]
        return pr + dr * 4, pc + dc * 4

    elif ghost_id == GhostID.INKY:
        # Flank — 2x vector from Blinky to 2 tiles ahead of Pac-Man
        dr, dc = DIRECTION_DELTAS[pac_dir]
        ahead_r, ahead_c = pr + dr * 2, pc + dc * 2
        blinky_r, blinky_c = int(ghost_pos[GhostID.BLINKY, 0]), int(ghost_pos[GhostID.BLINKY, 1])
        return ahead_r + (ahead_r - blinky_r), ahead_c + (ahead_c - blinky_c)

    elif ghost_id == GhostID.CLYDE:
        # Fickle — chase if far, scatter if close
        clyde_r, clyde_c = int(ghost_pos[GhostID.CLYDE, 0]), int(ghost_pos[GhostID.CLYDE, 1])
        dist = abs(pr - clyde_r) + abs(pc - clyde_c)
        if dist > 8:
            return pr, pc
        else:
            return SCATTER_TARGETS[GhostID.CLYDE]

    return pr, pc  # fallback


def compute_ghost_target(
    ghost_id: int,
    mode: int,
    pac_pos: np.ndarray,
    pac_dir: int,
    ghost_pos: np.ndarray,
    difficulty: int,
    pellets_remaining: int,
) -> tuple[int, int]:
    """Compute target tile for a ghost given its current mode."""
    if mode == GhostMode.SCATTER:
        # Cruise Elroy: Blinky chases even in scatter at high difficulty
        if (ghost_id == GhostID.BLINKY and difficulty >= 2
                and pellets_remaining <= ELROY_1_THRESHOLD):
            return compute_chase_target(ghost_id, pac_pos, pac_dir, ghost_pos)
        return SCATTER_TARGETS[GhostID(ghost_id)]

    elif mode == GhostMode.CHASE:
        return compute_chase_target(ghost_id, pac_pos, pac_dir, ghost_pos)

    # FRIGHTENED and EATEN handled elsewhere (random / return path)
    return SCATTER_TARGETS[GhostID(ghost_id)]


def choose_direction_toward_target(
    grid: np.ndarray,
    ghost_row: int,
    ghost_col: int,
    current_dir: int,
    target_row: int,
    target_col: int,
) -> int:
    """At an intersection, pick direction minimizing Euclidean distance to target.
    Cannot reverse. Ties broken by DIRECTION_PRIORITY (UP > LEFT > DOWN > RIGHT).
    """
    opposite = OPPOSITE_DIRECTION[Direction(current_dir)]
    best_dir = current_dir
    best_dist = float("inf")

    for d in DIRECTION_PRIORITY:
        if d == opposite:
            continue
        dr, dc = DIRECTION_DELTAS[d]
        nr, nc = ghost_row + dr, (ghost_col + dc) % MAZE_COLS
        if not is_walkable(grid, nr, nc, for_ghost=True):
            continue
        dist = (nr - target_row) ** 2 + (nc - target_col) ** 2
        if dist < best_dist:
            best_dist = dist
            best_dir = d

    return best_dir


def choose_frightened_direction(
    grid: np.ndarray,
    ghost_row: int,
    ghost_col: int,
    current_dir: int,
    rng: np.random.Generator,
) -> int:
    """Frightened mode: random legal direction (no reversal)."""
    opposite = OPPOSITE_DIRECTION[Direction(current_dir)]
    legal = []
    for d in Direction:
        if d == opposite:
            continue
        dr, dc = DIRECTION_DELTAS[d]
        nr, nc = ghost_row + dr, (ghost_col + dc) % MAZE_COLS
        if is_walkable(grid, nr, nc, for_ghost=True):
            legal.append(d)
    if not legal:
        return int(opposite)  # forced reversal if trapped
    return int(rng.choice(legal))
```

- [ ] **Step 2: Verify**

Run: `python -c "from pacman.engine.ghost_ai import compute_chase_target; import numpy as np; print(compute_chase_target(0, np.array([15,14]), 0, np.zeros((4,2),dtype=np.int16)))"`
Expected: `(15, 14)` (Blinky targets Pac-Man's exact position)

- [ ] **Step 3: Commit**

```bash
git add pacman/engine/ghost_ai.py
git commit -m "feat(v2): classic arcade ghost AI — Blinky/Pinky/Inky/Clyde behaviors"
```

---

### Task 5: Game Step Logic — `pacman/engine/game.py`

**Files:**
- Create: `pacman/engine/game.py`

- [ ] **Step 1: Write the core game step function**

```python
# pacman/engine/game.py
"""Single-game step logic — pure functions operating on GameState."""
import numpy as np

from .constants import (
    Direction, Tile, GhostMode, GhostID, DIRECTION_DELTAS, OPPOSITE_DIRECTION,
    MAZE_ROWS, MAZE_COLS, NUM_GHOSTS,
)
from .maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, GHOST_DOOR_POS, FRUIT_POSITION,
)
from .maze import is_walkable
from .entities import GameState, reset_positions
from . import ghost_ai


def get_legal_actions(grid: np.ndarray, pac_pos: np.ndarray) -> np.ndarray:
    """Return (4,) bool mask of legal Pac-Man moves."""
    row, col = int(pac_pos[0]), int(pac_pos[1])
    mask = np.zeros(4, dtype=bool)
    for d in Direction:
        dr, dc = DIRECTION_DELTAS[d]
        nr, nc = row + dr, (col + dc) % MAZE_COLS
        mask[d] = is_walkable(grid, nr, nc, for_ghost=False)
    return mask


def step_game(
    state: GameState,
    pac_action: int,
    config: dict,
    return_paths: np.ndarray,
    rng: np.random.Generator,
) -> tuple[GameState, list[str], float]:
    """Execute one game tick. Returns (state, events, reward).

    Mutates state in-place for performance. Events is a list of string event names.
    """
    events = []
    state.step_count += 1

    # --- 1. Move Pac-Man ---
    _move_pacman(state, pac_action)

    # --- 2. Check pickups ---
    _check_pickups(state, config, events)

    # --- 3. Check Pac-Man vs ghost collision (pre-ghost-move) ---
    if _check_collision(state, config, events):
        reward = compute_reward(events, config, state)
        return state, events, reward

    # --- 4. Move ghosts ---
    _move_ghosts(state, config, return_paths, rng)

    # --- 5. Check collision again (post-ghost-move) ---
    if _check_collision(state, config, events):
        reward = compute_reward(events, config, state)
        return state, events, reward

    # --- 6. Update mode timers ---
    _update_mode_timers(state, config)

    # --- 7. Update ghost house exits ---
    _update_ghost_house(state, config)

    # --- 8. Update power timer ---
    if state.pac_powered:
        state.pac_power_timer -= 1
        if state.pac_power_timer <= 0:
            state.pac_powered = False
            state.pac_ghosts_eaten = 0
            # Restore non-eaten ghosts to global mode
            for i in range(NUM_GHOSTS):
                if state.ghost_mode[i] == GhostMode.FRIGHTENED:
                    state.ghost_mode[i] = state.global_mode

    # --- 9. Fruit ---
    _update_fruit(state, config)

    # --- 10. Check win ---
    if state.pellets_remaining <= 0:
        state.done = True
        state.winner = "pacman"
        events.append("clear_level")

    # --- 11. Check timeout ---
    if not state.done and state.step_count >= config["game"]["max_steps"]:
        state.done = True
        events.append("timeout")

    reward = compute_reward(events, config, state)
    return state, events, reward


def _move_pacman(state: GameState, action: int) -> None:
    """Move Pac-Man in the desired direction if legal, else continue current."""
    row, col = int(state.pac_pos[0]), int(state.pac_pos[1])
    dr, dc = DIRECTION_DELTAS[action]
    nr, nc = row + dr, (col + dc) % MAZE_COLS

    if is_walkable(state.grid, nr, nc, for_ghost=False):
        state.pac_pos[0] = nr
        state.pac_pos[1] = nc
        state.pac_dir = action
    else:
        # Try continuing current direction
        dr, dc = DIRECTION_DELTAS[state.pac_dir]
        nr, nc = row + dr, (col + dc) % MAZE_COLS
        if is_walkable(state.grid, nr, nc, for_ghost=False):
            state.pac_pos[0] = nr
            state.pac_pos[1] = nc


def _check_pickups(state: GameState, config: dict, events: list) -> None:
    """Check if Pac-Man is on a pellet, power pellet, or fruit."""
    r, c = int(state.pac_pos[0]), int(state.pac_pos[1])
    tile = state.grid[r, c]

    if tile == Tile.PELLET:
        state.grid[r, c] = Tile.EMPTY
        state.pellets_eaten += 1
        state.pellets_remaining -= 1
        state.score += 10
        events.append("eat_pellet")

    elif tile == Tile.POWER_PELLET:
        state.grid[r, c] = Tile.EMPTY
        state.pellets_eaten += 1
        state.pellets_remaining -= 1
        state.score += 50
        state.pac_powered = True
        state.pac_power_timer = config["game"]["frightened_duration"]
        state.pac_ghosts_eaten = 0
        events.append("eat_power_pellet")
        # Frighten all active ghosts
        for i in range(NUM_GHOSTS):
            if (state.ghost_mode[i] not in (GhostMode.EATEN,)
                    and not state.ghost_in_house[i]):
                state.ghost_mode[i] = GhostMode.FRIGHTENED
                state.ghost_fright_timer[i] = config["game"]["frightened_duration"]
                state.ghost_dir[i] = OPPOSITE_DIRECTION[Direction(state.ghost_dir[i])]

    if (state.fruit_active
            and r == FRUIT_POSITION[0] and c == FRUIT_POSITION[1]):
        state.score += config["game"]["fruit"]["score"]
        state.fruit_active = False
        events.append("eat_fruit")


def _check_collision(state: GameState, config: dict, events: list) -> bool:
    """Check Pac-Man vs ghost collision. Returns True if episode should stop stepping."""
    pr, pc = int(state.pac_pos[0]), int(state.pac_pos[1])

    for i in range(NUM_GHOSTS):
        if state.ghost_in_house[i]:
            continue
        if state.ghost_mode[i] == GhostMode.EATEN:
            continue
        gr, gc = int(state.ghost_pos[i, 0]), int(state.ghost_pos[i, 1])
        if pr != gr or pc != gc:
            continue

        if state.ghost_mode[i] == GhostMode.FRIGHTENED:
            # Pac-Man eats ghost
            state.ghost_mode[i] = GhostMode.EATEN
            ghost_scores = [200, 400, 800, 1600]
            idx = min(state.pac_ghosts_eaten, 3)
            state.score += ghost_scores[idx]
            state.pac_ghosts_eaten += 1
            events.append(f"eat_ghost_{state.pac_ghosts_eaten}")
        else:
            # Ghost catches Pac-Man
            state.pac_lives -= 1
            events.append("death")
            if state.pac_lives <= 0:
                state.done = True
                state.winner = "ghosts"
                events.append("game_over")
            else:
                reset_positions(state, config)
            return True  # stop further processing this tick

    return False


def _move_ghosts(
    state: GameState,
    config: dict,
    return_paths: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Move all ghosts according to their current mode."""
    for i in range(NUM_GHOSTS):
        gr, gc = int(state.ghost_pos[i, 0]), int(state.ghost_pos[i, 1])

        # Ghost in house — move toward door
        if state.ghost_in_house[i]:
            if state.ghost_exiting[i]:
                _move_ghost_exit_house(state, i)
            continue

        mode = state.ghost_mode[i]

        if mode == GhostMode.EATEN:
            # Follow pre-computed return path
            door_r, door_c = GHOST_DOOR_POS
            if gr == door_r and gc == door_c:
                # Reached door — re-enter house and respawn
                state.ghost_mode[i] = state.global_mode
                state.ghost_fright_timer[i] = 0
                state.ghost_in_house[i] = True
                state.ghost_exiting[i] = True
                continue
            direction = return_paths[gr, gc]
            if direction >= 0:
                dr, dc = DIRECTION_DELTAS[direction]
                state.ghost_pos[i, 0] = gr + dr
                state.ghost_pos[i, 1] = (gc + dc) % MAZE_COLS
                state.ghost_dir[i] = direction

        elif mode == GhostMode.FRIGHTENED:
            new_dir = ghost_ai.choose_frightened_direction(
                state.grid, gr, gc, int(state.ghost_dir[i]), rng,
            )
            dr, dc = DIRECTION_DELTAS[new_dir]
            state.ghost_pos[i, 0] = gr + dr
            state.ghost_pos[i, 1] = (gc + dc) % MAZE_COLS
            state.ghost_dir[i] = new_dir

        else:  # SCATTER or CHASE
            target = ghost_ai.compute_ghost_target(
                i, mode, state.pac_pos, state.pac_dir,
                state.ghost_pos, state.difficulty, state.pellets_remaining,
            )
            new_dir = ghost_ai.choose_direction_toward_target(
                state.grid, gr, gc, int(state.ghost_dir[i]),
                target[0], target[1],
            )
            dr, dc = DIRECTION_DELTAS[new_dir]
            state.ghost_pos[i, 0] = gr + dr
            state.ghost_pos[i, 1] = (gc + dc) % MAZE_COLS
            state.ghost_dir[i] = new_dir


def _move_ghost_exit_house(state: GameState, ghost_idx: int) -> None:
    """Move ghost upward out of the ghost house."""
    door_r, door_c = GHOST_DOOR_POS
    gr = int(state.ghost_pos[ghost_idx, 0])
    gc = int(state.ghost_pos[ghost_idx, 1])

    # Move toward door column first, then up to door row
    if gc < door_c:
        state.ghost_pos[ghost_idx, 1] = gc + 1
    elif gc > door_c:
        state.ghost_pos[ghost_idx, 1] = gc - 1
    elif gr > door_r:
        state.ghost_pos[ghost_idx, 0] = gr - 1
    else:
        # Reached exit — now outside
        state.ghost_in_house[ghost_idx] = False
        state.ghost_exiting[ghost_idx] = False
        state.ghost_pos[ghost_idx, 0] = door_r - 1  # one above door
        state.ghost_mode[ghost_idx] = state.global_mode


def _update_mode_timers(state: GameState, config: dict) -> None:
    """Update scatter/chase mode schedule timers."""
    # Decrement individual frightened timers
    for i in range(NUM_GHOSTS):
        if state.ghost_mode[i] == GhostMode.FRIGHTENED:
            state.ghost_fright_timer[i] -= 1
            if state.ghost_fright_timer[i] <= 0:
                state.ghost_mode[i] = state.global_mode

    # Difficulty 0: scatter only, no mode schedule
    if state.difficulty == 0:
        return

    schedule = config["game"]["mode_schedule"]
    if state.mode_index >= len(schedule):
        return
    if schedule[state.mode_index] == -1:
        return  # permanent mode

    state.mode_timer -= 1
    if state.mode_timer <= 0:
        state.mode_index += 1
        if state.mode_index < len(schedule):
            state.mode_timer = schedule[state.mode_index]
            # Toggle global mode
            if state.global_mode == GhostMode.SCATTER:
                state.global_mode = GhostMode.CHASE
            else:
                state.global_mode = GhostMode.SCATTER
            # Apply to non-frightened, non-eaten ghosts + reverse direction
            for i in range(NUM_GHOSTS):
                if state.ghost_mode[i] in (GhostMode.SCATTER, GhostMode.CHASE):
                    state.ghost_mode[i] = state.global_mode
                    state.ghost_dir[i] = OPPOSITE_DIRECTION[Direction(state.ghost_dir[i])]


def _update_ghost_house(state: GameState, config: dict) -> None:
    """Check if ghosts should exit the house based on pellets eaten."""
    thresholds = config["game"]["ghost_exit_pellets"]
    for i in range(NUM_GHOSTS):
        if state.ghost_in_house[i] and not state.ghost_exiting[i]:
            if state.pellets_eaten >= thresholds[i]:
                state.ghost_exiting[i] = True


def _update_fruit(state: GameState, config: dict) -> None:
    """Spawn/despawn fruit based on pellet thresholds."""
    if state.fruit_active:
        state.fruit_timer -= 1
        if state.fruit_timer <= 0:
            state.fruit_active = False
    else:
        for threshold in config["game"]["fruit"]["spawn_pellets"]:
            if (threshold not in state.fruit_spawned
                    and state.pellets_eaten >= threshold):
                state.fruit_active = True
                state.fruit_timer = config["game"]["fruit"]["duration"]
                state.fruit_spawned.append(threshold)
                break


def compute_reward(events: list[str], config: dict, state: GameState) -> float:
    """Map event list to scalar reward per spec section 8."""
    reward_cfg = config["rewards"]
    reward = reward_cfg["time_step"]

    for event in events:
        if event == "eat_pellet":
            reward += reward_cfg["eat_pellet"]
        elif event == "eat_power_pellet":
            reward += reward_cfg["eat_power_pellet"]
        elif event.startswith("eat_ghost_"):
            idx = int(event.split("_")[-1]) - 1  # 1-indexed
            idx = min(idx, len(reward_cfg["eat_ghost"]) - 1)
            reward += reward_cfg["eat_ghost"][idx]
        elif event == "eat_fruit":
            reward += reward_cfg["eat_fruit"]
        elif event == "clear_level":
            reward += reward_cfg["clear_level"]
        elif event == "death":
            reward += reward_cfg["death"]
        elif event == "game_over":
            reward += reward_cfg["game_over"]

    return reward
```

- [ ] **Step 2: Smoke test — run a random game**

```python
# Quick verification script (don't save, just run)
python -c "
from pacman.utils.config import load_config
from pacman.engine.entities import create_initial_state
from pacman.engine.game import step_game, get_legal_actions
from pacman.engine.maze import compute_ghost_return_paths
import numpy as np

config = load_config()
state = create_initial_state(config)
rp = compute_ghost_return_paths(state.grid)
rng = np.random.default_rng(42)

for _ in range(100):
    legal = get_legal_actions(state.grid, state.pac_pos)
    action = rng.choice(np.where(legal)[0])
    state, events, reward = step_game(state, action, config, rp, rng)
    if state.done:
        break

print(f'Steps: {state.step_count}, Score: {state.score}, Done: {state.done}, Winner: {state.winner}')
"
```

- [ ] **Step 3: Commit**

```bash
git add pacman/engine/game.py
git commit -m "feat(v2): game step logic — movement, collision, modes, rewards"
```

---

### Task 6: Engine Tests — `tests/test_engine.py`

**Files:**
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write comprehensive engine tests**

```python
# tests/test_engine.py
import numpy as np
import pytest

from pacman.utils.config import load_config
from pacman.engine.constants import (
    Direction, Tile, GhostMode, GhostID, MAZE_ROWS, MAZE_COLS, NUM_GHOSTS,
)
from pacman.engine.maze import (
    load_initial_grid, count_pellets, count_power_pellets,
    is_walkable, get_legal_directions, compute_ghost_return_paths,
)
from pacman.engine.maze_data import PACMAN_START, GHOST_DOOR_POS
from pacman.engine.entities import create_initial_state, reset_positions
from pacman.engine.game import step_game, get_legal_actions, compute_reward
from pacman.engine import ghost_ai


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def grid():
    return load_initial_grid()


@pytest.fixture
def state(config):
    return create_initial_state(config)


@pytest.fixture
def return_paths(grid):
    return compute_ghost_return_paths(grid)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# --- Maze Tests ---

class TestMaze:
    def test_grid_dimensions(self, grid):
        assert grid.shape == (MAZE_ROWS, MAZE_COLS)
        assert grid.dtype == np.int8

    def test_corners_are_walls(self, grid):
        assert grid[0, 0] == Tile.WALL
        assert grid[0, MAZE_COLS - 1] == Tile.WALL
        assert grid[MAZE_ROWS - 1, 0] == Tile.WALL
        assert grid[MAZE_ROWS - 1, MAZE_COLS - 1] == Tile.WALL

    def test_pellet_count(self, grid):
        total = count_pellets(grid)
        power = count_power_pellets(grid)
        assert total > 200  # classic maze has ~244
        assert power == 4

    def test_pacman_start_is_walkable(self, grid):
        r, c = PACMAN_START
        assert is_walkable(grid, r, c, for_ghost=False)

    def test_wall_not_walkable(self, grid):
        assert not is_walkable(grid, 0, 0, for_ghost=False)

    def test_out_of_bounds(self, grid):
        assert not is_walkable(grid, -1, 0)
        assert not is_walkable(grid, MAZE_ROWS, 0)

    def test_legal_directions_at_start(self, grid):
        r, c = PACMAN_START
        dirs = get_legal_directions(grid, r, c)
        assert len(dirs) >= 1

    def test_return_paths_shape(self, return_paths):
        assert return_paths.shape == (MAZE_ROWS, MAZE_COLS)
        assert return_paths.dtype == np.int8

    def test_return_paths_at_door(self, return_paths):
        r, c = GHOST_DOOR_POS
        assert return_paths[r, c] >= 0  # should have a direction (or 0 as at-destination)

    def test_return_paths_walkable_cells_reachable(self, grid, return_paths):
        """Most walkable cells should have a valid return direction."""
        walkable = (grid != Tile.WALL)
        reachable = return_paths >= 0
        # At least 90% of walkable cells should be reachable
        assert np.sum(walkable & reachable) > 0.9 * np.sum(walkable)


# --- Entity Tests ---

class TestEntities:
    def test_initial_state(self, state):
        assert state.pac_lives == 3
        assert state.score == 0
        assert state.step_count == 0
        assert not state.done
        assert state.winner is None
        assert state.pellets_remaining > 0
        assert state.pellets_remaining == state.total_pellets

    def test_ghost_positions(self, state):
        assert state.ghost_pos.shape == (NUM_GHOSTS, 2)
        # Blinky starts outside ghost house
        assert not state.ghost_in_house[GhostID.BLINKY]
        # Others start inside
        assert state.ghost_in_house[GhostID.PINKY]
        assert state.ghost_in_house[GhostID.INKY]
        assert state.ghost_in_house[GhostID.CLYDE]

    def test_reset_positions(self, state, config):
        state.pac_pos[:] = [0, 0]
        reset_positions(state, config)
        assert tuple(state.pac_pos) == PACMAN_START


# --- Ghost AI Tests ---

class TestGhostAI:
    def test_blinky_targets_pacman(self):
        pac_pos = np.array([15, 14], dtype=np.int16)
        ghost_pos = np.zeros((4, 2), dtype=np.int16)
        target = ghost_ai.compute_chase_target(GhostID.BLINKY, pac_pos, Direction.LEFT, ghost_pos)
        assert target == (15, 14)

    def test_pinky_targets_ahead(self):
        pac_pos = np.array([15, 14], dtype=np.int16)
        ghost_pos = np.zeros((4, 2), dtype=np.int16)
        target = ghost_ai.compute_chase_target(GhostID.PINKY, pac_pos, Direction.LEFT, ghost_pos)
        assert target == (15, 10)  # 4 tiles left

    def test_clyde_retreats_when_close(self):
        pac_pos = np.array([15, 14], dtype=np.int16)
        ghost_pos = np.zeros((4, 2), dtype=np.int16)
        ghost_pos[GhostID.CLYDE] = [15, 16]  # 2 tiles away
        target = ghost_ai.compute_chase_target(GhostID.CLYDE, pac_pos, Direction.LEFT, ghost_pos)
        # Should retreat to scatter corner when close
        from pacman.engine.constants import SCATTER_TARGETS
        assert target == SCATTER_TARGETS[GhostID.CLYDE]


# --- Game Step Tests ---

class TestGame:
    def test_step_increments_counter(self, state, config, return_paths, rng):
        step_game(state, Direction.LEFT, config, return_paths, rng)
        assert state.step_count == 1

    def test_legal_actions_not_empty(self, state):
        legal = get_legal_actions(state.grid, state.pac_pos)
        assert legal.any()

    def test_random_game_completes(self, config, return_paths):
        """Run a full random game — should terminate without errors."""
        state = create_initial_state(config)
        rng = np.random.default_rng(123)
        for _ in range(config["game"]["max_steps"]):
            legal = get_legal_actions(state.grid, state.pac_pos)
            action = rng.choice(np.where(legal)[0])
            step_game(state, int(action), config, return_paths, rng)
            if state.done:
                break
        assert state.done

    def test_pellets_decrease(self, state, config, return_paths, rng):
        initial = state.pellets_remaining
        for _ in range(200):
            legal = get_legal_actions(state.grid, state.pac_pos)
            action = rng.choice(np.where(legal)[0])
            step_game(state, int(action), config, return_paths, rng)
            if state.done:
                break
        # After 200 steps, some pellets should have been eaten
        assert state.pellets_remaining <= initial

    def test_timeout(self, config, return_paths, rng):
        config = dict(config)
        config["game"] = dict(config["game"])
        config["game"]["max_steps"] = 5
        state = create_initial_state(config)
        for _ in range(10):
            step_game(state, Direction.LEFT, config, return_paths, rng)
            if state.done:
                break
        assert state.done
        assert state.step_count <= 5

    def test_compute_reward(self, config, state):
        r = compute_reward(["eat_pellet"], config, state)
        assert r == config["rewards"]["eat_pellet"] + config["rewards"]["time_step"]

        r = compute_reward(["clear_level"], config, state)
        assert r == config["rewards"]["clear_level"] + config["rewards"]["time_step"]
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_engine.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine.py
git commit -m "test(v2): comprehensive engine tests — maze, entities, ghost AI, game step"
```

---

### Task 7: Single Environment — `pacman/env/pacman_env.py`

**Files:**
- Create: `pacman/env/pacman_env.py`

- [ ] **Step 1: Write Gymnasium-compatible single environment**

```python
# pacman/env/pacman_env.py
"""Single Pac-Man environment with Gymnasium-compatible interface."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, MAZE_ROWS, MAZE_COLS, NUM_GHOSTS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths
from ..engine.maze_data import FRUIT_POSITION
from ..engine.entities import create_initial_state, GameState
from ..engine.game import step_game, get_legal_actions

NUM_CHANNELS = 8
NUM_SCALARS = 4


class PacmanEnv:
    """Single Pac-Man environment for evaluation and visualization."""

    def __init__(self, config: dict, difficulty: int = 0):
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._state: GameState | None = None
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = create_initial_state(self.config, self.difficulty)
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        state, events, reward = step_game(
            self._state, action, self.config, self._return_paths, self._rng,
        )
        self._state = state
        obs = self._build_obs()
        terminated = state.done
        truncated = False
        info = {
            "score": state.score,
            "pellets_eaten": state.pellets_eaten,
            "lives": state.pac_lives,
            "winner": state.winner,
            "events": events,
        }
        return obs, reward, terminated, truncated, info

    def get_legal_mask(self) -> np.ndarray:
        return get_legal_actions(self._state.grid, self._state.pac_pos)

    @property
    def state(self) -> GameState:
        return self._state

    def _build_obs(self) -> dict:
        """Build 8-channel grid + 4 scalars observation."""
        s = self._state
        grid = np.zeros((NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)

        # Ch 0: Walls
        grid[0] = (s.grid == Tile.WALL).astype(np.float32)
        # Ch 1: Pac-Man position
        grid[1, s.pac_pos[0], s.pac_pos[1]] = 1.0
        # Ch 2: Pellets
        grid[2] = (s.grid == Tile.PELLET).astype(np.float32)
        # Ch 3: Power pellets
        grid[3] = (s.grid == Tile.POWER_PELLET).astype(np.float32)
        # Ch 4: Dangerous ghosts (scatter/chase)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] in (GhostMode.SCATTER, GhostMode.CHASE):
                grid[4, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0
        # Ch 5: Edible ghosts (frightened)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] == GhostMode.FRIGHTENED:
                grid[5, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0
        # Ch 6: Ghost house
        grid[6] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR)).astype(np.float32)
        # Ch 7: Fruit
        if s.fruit_active:
            grid[7, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

        # Scalars
        max_fright = self.config["game"]["frightened_duration"]
        scalars = np.array([
            s.pac_power_timer / max(max_fright, 1),
            s.pac_lives / self.config["game"]["lives"],
            s.pac_ghosts_eaten / 4.0,
            s.pellets_eaten / max(s.total_pellets, 1),
        ], dtype=np.float32)

        return {"grid": grid, "scalars": scalars}
```

- [ ] **Step 2: Verify**

Run: `python -c "from pacman.utils.config import load_config; from pacman.env.pacman_env import PacmanEnv; env = PacmanEnv(load_config()); obs, _ = env.reset(seed=42); print(obs['grid'].shape, obs['scalars'].shape)"`
Expected: `(8, 31, 28) (4,)`

- [ ] **Step 3: Commit**

```bash
git add pacman/env/pacman_env.py
git commit -m "feat(v2): single Pac-Man environment with CNN observation builder"
```

---

### Task 8: Vectorized Environment — `pacman/env/vec_env.py` + Tests

**Files:**
- Create: `pacman/env/vec_env.py`
- Create: `tests/test_vec_env.py`

- [ ] **Step 1: Write vectorized environment**

```python
# pacman/env/vec_env.py
"""Vectorized Pac-Man environment — N games as batched NumPy operations."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, GhostID, Direction, DIRECTION_DELTAS, OPPOSITE_DIRECTION,
    MAZE_ROWS, MAZE_COLS, NUM_GHOSTS, NUM_ACTIONS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths
from ..engine.maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, FRUIT_POSITION,
)
from ..engine.entities import create_initial_state
from ..engine.game import step_game, get_legal_actions
from .pacman_env import NUM_CHANNELS, NUM_SCALARS


class VecEnv:
    """Vectorized Pac-Man environment stepping N games in parallel.

    Uses per-game step_game() internally with auto-reset.
    A fully batched NumPy implementation can be done as a future optimization.
    """

    def __init__(self, num_envs: int, config: dict, difficulty: int = 0):
        self.num_envs = num_envs
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._states = []
        self._rngs = []

    def reset(self, seed: int | None = None) -> dict:
        base_seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._states = []
        self._rngs = []
        for i in range(self.num_envs):
            self._states.append(create_initial_state(self.config, self.difficulty))
            self._rngs.append(np.random.default_rng(base_seed + i))
        return self._build_batch_obs()

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """Step all environments. Auto-resets done envs.

        Returns: (obs_dict, rewards, dones, infos)
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = {
            "score": np.zeros(self.num_envs, dtype=np.int32),
            "pellets_eaten": np.zeros(self.num_envs, dtype=np.int32),
            "lives": np.zeros(self.num_envs, dtype=np.int32),
            "winner": [None] * self.num_envs,
            "level_cleared": np.zeros(self.num_envs, dtype=bool),
        }

        for i in range(self.num_envs):
            state, events, reward = step_game(
                self._states[i], int(actions[i]),
                self.config, self._return_paths, self._rngs[i],
            )
            rewards[i] = reward
            dones[i] = state.done
            infos["score"][i] = state.score
            infos["pellets_eaten"][i] = state.pellets_eaten
            infos["lives"][i] = state.pac_lives
            infos["winner"][i] = state.winner
            infos["level_cleared"][i] = state.winner == "pacman"

            # Auto-reset done environments
            if state.done:
                self._states[i] = create_initial_state(self.config, self.difficulty)

        obs = self._build_batch_obs()
        return obs, rewards, dones, infos

    def get_legal_masks(self) -> np.ndarray:
        """Return (N, 4) bool mask of legal actions per env."""
        masks = np.zeros((self.num_envs, NUM_ACTIONS), dtype=bool)
        for i in range(self.num_envs):
            masks[i] = get_legal_actions(self._states[i].grid, self._states[i].pac_pos)
        return masks

    def set_difficulty(self, difficulty: int) -> None:
        self.difficulty = difficulty
        for state in self._states:
            state.difficulty = difficulty

    def _build_batch_obs(self) -> dict:
        """Build batched observations: grid (N,8,31,28), scalars (N,4)."""
        grids = np.zeros((self.num_envs, NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)
        scalars = np.zeros((self.num_envs, NUM_SCALARS), dtype=np.float32)
        max_fright = self.config["game"]["frightened_duration"]

        for i, s in enumerate(self._states):
            grids[i, 0] = (s.grid == Tile.WALL)
            grids[i, 1, s.pac_pos[0], s.pac_pos[1]] = 1.0
            grids[i, 2] = (s.grid == Tile.PELLET)
            grids[i, 3] = (s.grid == Tile.POWER_PELLET)
            for g in range(NUM_GHOSTS):
                if not s.ghost_in_house[g] and s.ghost_mode[g] in (GhostMode.SCATTER, GhostMode.CHASE):
                    grids[i, 4, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0
                if not s.ghost_in_house[g] and s.ghost_mode[g] == GhostMode.FRIGHTENED:
                    grids[i, 5, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0
            grids[i, 6] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR))
            if s.fruit_active:
                grids[i, 7, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

            scalars[i] = [
                s.pac_power_timer / max(max_fright, 1),
                s.pac_lives / self.config["game"]["lives"],
                s.pac_ghosts_eaten / 4.0,
                s.pellets_eaten / max(s.total_pellets, 1),
            ]

        return {"grid": grids, "scalars": scalars}
```

- [ ] **Step 2: Write vec env tests**

```python
# tests/test_vec_env.py
import numpy as np
import pytest

from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.env.vec_env import VecEnv


@pytest.fixture
def config():
    return load_config()


class TestVecEnv:
    def test_observation_shapes(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert obs["scalars"].shape == (4, 4)
        assert obs["grid"].dtype == np.float32
        assert obs["scalars"].dtype == np.float32

    def test_observation_values_in_range(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].min() >= 0.0
        assert obs["grid"].max() <= 1.0
        assert obs["scalars"].min() >= 0.0
        assert obs["scalars"].max() <= 1.0

    def test_legal_masks_shape(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        assert masks.shape == (4, 4)
        assert masks.dtype == bool
        assert masks.any(axis=1).all()  # every env has at least 1 legal action

    def test_step_returns_correct_shapes(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        actions = np.array([np.random.choice(np.where(m)[0]) for m in masks])
        obs, rewards, dones, infos = env.step(actions)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_auto_reset(self, config):
        """Run until at least one env finishes, verify it auto-resets."""
        config = dict(config)
        config["game"] = dict(config["game"])
        config["game"]["max_steps"] = 50  # force quick termination
        env = VecEnv(4, config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        seen_done = False
        for _ in range(100):
            masks = env.get_legal_masks()
            actions = np.array([rng.choice(np.where(m)[0]) for m in masks])
            obs, rewards, dones, infos = env.step(actions)
            if dones.any():
                seen_done = True
                # After auto-reset, obs should still be valid
                assert obs["grid"].shape == (4, 8, 31, 28)
                break
        assert seen_done

    def test_parity_with_single_env(self, config):
        """VecEnv(N=1) should produce identical results to PacmanEnv with same seed."""
        single = PacmanEnv(config, difficulty=0)
        vec = VecEnv(1, config, difficulty=0)

        single_obs, _ = single.reset(seed=100)
        vec_obs = vec.reset(seed=100)

        np.testing.assert_array_equal(single_obs["grid"], vec_obs["grid"][0])
        np.testing.assert_array_equal(single_obs["scalars"], vec_obs["scalars"][0])

        rng_single = np.random.default_rng(200)
        rng_vec = np.random.default_rng(200)
        for _ in range(50):
            mask_s = single.get_legal_mask()
            mask_v = vec.get_legal_masks()[0]
            np.testing.assert_array_equal(mask_s, mask_v)

            action = rng_single.choice(np.where(mask_s)[0])
            obs_s, r_s, term_s, _, _ = single.step(int(action))
            obs_v, r_v, d_v, _ = vec.step(np.array([action]))

            np.testing.assert_array_almost_equal(r_s, r_v[0])
            if term_s:
                break

    def test_set_difficulty(self, config):
        env = VecEnv(2, config, difficulty=0)
        env.reset(seed=42)
        env.set_difficulty(2)
        assert env.difficulty == 2
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_vec_env.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pacman/env/vec_env.py tests/test_vec_env.py
git commit -m "feat(v2): vectorized environment + tests — N parallel games with auto-reset"
```

---

### Task 9: CNN Network — `pacman/agents/networks.py` + Tests

**Files:**
- Create: `pacman/agents/networks.py`
- Create: `tests/test_networks.py`

- [ ] **Step 1: Write actor-critic network**

```python
# pacman/agents/networks.py
"""CNN Actor-Critic network for Pac-Man PPO."""
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared CNN backbone with separate policy and value heads."""

    def __init__(
        self,
        grid_channels: int = 8,
        grid_h: int = 31,
        grid_w: int = 28,
        num_scalars: int = 4,
        num_actions: int = 4,
        cnn_channels: list[int] = (32, 64, 64),
        cnn_kernels: list[int] = (3, 3, 3),
        cnn_strides: list[int] = (1, 2, 2),
        shared_hidden: int = 512,
        head_hidden: int = 128,
    ):
        super().__init__()
        # CNN encoder
        layers = []
        in_ch = grid_channels
        for out_ch, k, s in zip(cnn_channels, cnn_kernels, cnn_strides):
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_h, grid_w)
            cnn_out = self.cnn(dummy).view(1, -1).shape[1]

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(cnn_out + num_scalars, shared_hidden),
            nn.ReLU(),
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Policy head final layer: small init for uniform exploration
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        # Value head final layer: standard init
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(
        self,
        grid: torch.Tensor,      # (N, 8, 31, 28)
        scalars: torch.Tensor,    # (N, 4)
        legal_mask: torch.Tensor, # (N, 4) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        cnn_out = self.cnn(grid).flatten(1)
        combined = torch.cat([cnn_out, scalars], dim=1)
        shared = self.shared(combined)
        logits = self.policy(shared)
        # Mask illegal actions
        logits = logits.masked_fill(~legal_mask, -1e8)
        value = self.value(shared)
        return logits, value
```

- [ ] **Step 2: Write network tests**

```python
# tests/test_networks.py
import torch
import pytest

from pacman.agents.networks import ActorCritic


@pytest.fixture
def network():
    return ActorCritic()


class TestActorCritic:
    def test_forward_shapes(self, network):
        grid = torch.randn(2, 8, 31, 28)
        scalars = torch.randn(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits, value = network(grid, scalars, mask)
        assert logits.shape == (2, 4)
        assert value.shape == (2, 1)

    def test_softmax_sums_to_one(self, network):
        grid = torch.randn(3, 8, 31, 28)
        scalars = torch.randn(3, 4)
        mask = torch.ones(3, 4, dtype=torch.bool)
        logits, _ = network(grid, scalars, mask)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_action_masking(self, network):
        grid = torch.randn(1, 8, 31, 28)
        scalars = torch.randn(1, 4)
        mask = torch.tensor([[True, False, False, True]])
        logits, _ = network(grid, scalars, mask)
        probs = torch.softmax(logits, dim=-1)
        # Masked actions should have near-zero probability
        assert probs[0, 1].item() < 1e-6
        assert probs[0, 2].item() < 1e-6

    def test_parameter_count(self, network):
        total = sum(p.numel() for p in network.parameters())
        # Should be approximately 1.65M
        assert 1_000_000 < total < 3_000_000

    def test_gradients_flow(self, network):
        grid = torch.randn(2, 8, 31, 28)
        scalars = torch.randn(2, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits, value = network(grid, scalars, mask)
        loss = logits.sum() + value.sum()
        loss.backward()
        for p in network.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_networks.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pacman/agents/networks.py tests/test_networks.py
git commit -m "feat(v2): CNN actor-critic network (~1.65M params) + tests"
```

---

### Task 10: Rollout Buffer — `pacman/agents/rollout.py`

**Files:**
- Create: `pacman/agents/rollout.py`

- [ ] **Step 1: Write on-policy rollout buffer**

```python
# pacman/agents/rollout.py
"""On-policy rollout buffer for PPO."""
import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(
        self,
        num_envs: int,
        rollout_steps: int,
        grid_shape: tuple[int, int, int],  # (C, H, W)
        num_scalars: int,
        num_actions: int,
    ):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.total = num_envs * rollout_steps

        T, N = rollout_steps, num_envs
        self.obs_grids = np.zeros((T, N, *grid_shape), dtype=np.float32)
        self.obs_scalars = np.zeros((T, N, num_scalars), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)
        self.legal_masks = np.zeros((T, N, num_actions), dtype=bool)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

        self._step = 0

    def insert(
        self,
        obs_grid: np.ndarray,
        obs_scalars: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        t = self._step
        self.obs_grids[t] = obs_grid
        self.obs_scalars[t] = obs_scalars
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done
        self.legal_masks[t] = legal_mask
        self._step += 1

    def compute_gae(
        self,
        last_value: np.ndarray,  # (N,)
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns in-place."""
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def batch_generator(self, minibatch_size: int, device: torch.device):
        """Yield shuffled minibatches as torch tensors."""
        T, N = self.rollout_steps, self.num_envs
        total = T * N
        indices = np.random.permutation(total)

        # Flatten (T, N, ...) -> (T*N, ...)
        flat_grids = self.obs_grids.reshape(total, *self.obs_grids.shape[2:])
        flat_scalars = self.obs_scalars.reshape(total, -1)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_masks = self.legal_masks.reshape(total, -1)
        flat_values = self.values.reshape(total)

        for start in range(0, total, minibatch_size):
            end = start + minibatch_size
            idx = indices[start:end]
            yield {
                "obs_grid": torch.as_tensor(flat_grids[idx], device=device),
                "obs_scalars": torch.as_tensor(flat_scalars[idx], device=device),
                "actions": torch.as_tensor(flat_actions[idx], device=device),
                "old_log_probs": torch.as_tensor(flat_log_probs[idx], device=device),
                "advantages": torch.as_tensor(flat_advantages[idx], device=device),
                "returns": torch.as_tensor(flat_returns[idx], device=device),
                "legal_masks": torch.as_tensor(flat_masks[idx], device=device),
                "old_values": torch.as_tensor(flat_values[idx], device=device),
            }

    def reset(self) -> None:
        self._step = 0
```

- [ ] **Step 2: Commit**

```bash
git add pacman/agents/rollout.py
git commit -m "feat(v2): on-policy rollout buffer with GAE computation"
```

---

### Task 11: PPO Algorithm — `pacman/agents/ppo.py` + Tests

**Files:**
- Create: `pacman/agents/ppo.py`
- Create: `tests/test_ppo.py`

- [ ] **Step 1: Write PPO agent**

```python
# pacman/agents/ppo.py
"""Proximal Policy Optimization with clipped surrogate objective."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .networks import ActorCritic
from .rollout import RolloutBuffer


class PPO:
    """PPO agent for Pac-Man."""

    def __init__(self, network: ActorCritic, config: dict, device: torch.device):
        self.network = network.to(device)
        self.device = device
        self.config = config["ppo"]

        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=self.config["learning_rate"], eps=1e-5,
        )

    @torch.no_grad()
    def select_action(
        self,
        obs_grid: np.ndarray,     # (N, 8, 31, 28)
        obs_scalars: np.ndarray,  # (N, 4)
        legal_masks: np.ndarray,  # (N, 4) bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for all environments. Returns (actions, log_probs, values)."""
        grid_t = torch.as_tensor(obs_grid, device=self.device)
        scalars_t = torch.as_tensor(obs_scalars, device=self.device)
        mask_t = torch.as_tensor(legal_masks, device=self.device)

        logits, values = self.network(grid_t, scalars_t, mask_t)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    @torch.no_grad()
    def get_value(self, obs_grid: np.ndarray, obs_scalars: np.ndarray,
                  legal_masks: np.ndarray) -> np.ndarray:
        """Get value estimate for terminal bootstrap."""
        grid_t = torch.as_tensor(obs_grid, device=self.device)
        scalars_t = torch.as_tensor(obs_scalars, device=self.device)
        mask_t = torch.as_tensor(legal_masks, device=self.device)
        _, values = self.network(grid_t, scalars_t, mask_t)
        return values.squeeze(-1).cpu().numpy()

    def update(self, rollout: RolloutBuffer) -> dict:
        """Run PPO update over the rollout buffer. Returns loss metrics."""
        cfg = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Normalize advantages
        adv = rollout.advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rollout.advantages = adv.reshape(rollout.rollout_steps, rollout.num_envs)

        for _epoch in range(cfg["num_epochs"]):
            for batch in rollout.batch_generator(cfg["minibatch_size"], self.device):
                logits, values = self.network(
                    batch["obs_grid"], batch["obs_scalars"], batch["legal_masks"],
                )
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy()

                # Policy loss — clipped surrogate
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                advantages = batch["advantages"]
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"]) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss — clipped
                values_squeezed = values.squeeze(-1)
                value_pred_clipped = batch["old_values"] + torch.clamp(
                    values_squeezed - batch["old_values"],
                    -cfg["clip_epsilon"], cfg["clip_epsilon"],
                )
                value_loss1 = (values_squeezed - batch["returns"]) ** 2
                value_loss2 = (value_pred_clipped - batch["returns"]) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + cfg["value_loss_coef"] * value_loss
                    - self._entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def anneal_lr(self, update: int, total_updates: int) -> None:
        if not self.config.get("lr_anneal", True):
            return
        frac = 1.0 - update / total_updates
        lr = self.config["learning_rate"] * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def _entropy_coef(self) -> float:
        """Current entropy coefficient (set externally by trainer)."""
        return getattr(self, "_current_entropy_coef", self.config["entropy_coef_start"])

    def set_entropy_coef(self, coef: float) -> None:
        self._current_entropy_coef = coef
```

- [ ] **Step 2: Write PPO tests**

```python
# tests/test_ppo.py
import numpy as np
import torch
import pytest

from pacman.agents.networks import ActorCritic
from pacman.agents.rollout import RolloutBuffer
from pacman.agents.ppo import PPO
from pacman.utils.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def network():
    return ActorCritic()


@pytest.fixture
def ppo(network, config, device):
    return PPO(network, config, device)


class TestRolloutBuffer:
    def test_insert_and_shape(self):
        buf = RolloutBuffer(
            num_envs=4, rollout_steps=8,
            grid_shape=(8, 31, 28), num_scalars=4, num_actions=4,
        )
        for _ in range(8):
            buf.insert(
                obs_grid=np.random.randn(4, 8, 31, 28).astype(np.float32),
                obs_scalars=np.random.randn(4, 4).astype(np.float32),
                action=np.random.randint(0, 4, size=4),
                log_prob=np.random.randn(4).astype(np.float32),
                value=np.random.randn(4).astype(np.float32),
                reward=np.random.randn(4).astype(np.float32),
                done=np.zeros(4, dtype=bool),
                legal_mask=np.ones((4, 4), dtype=bool),
            )
        assert buf._step == 8

    def test_gae_computation(self):
        """Hand-verified GAE example with 3 steps, 1 env."""
        buf = RolloutBuffer(1, 3, (8, 31, 28), 4, 4)
        # Step 0: r=1, v=0.5, done=False
        # Step 1: r=2, v=1.0, done=False
        # Step 2: r=3, v=1.5, done=False
        # last_value = 2.0
        for t, (r, v) in enumerate([(1, 0.5), (2, 1.0), (3, 1.5)]):
            buf.insert(
                np.zeros((1, 8, 31, 28), dtype=np.float32),
                np.zeros((1, 4), dtype=np.float32),
                np.array([0]),
                np.array([0.0], dtype=np.float32),
                np.array([v], dtype=np.float32),
                np.array([r], dtype=np.float32),
                np.array([False]),
                np.ones((1, 4), dtype=bool),
            )
        buf.compute_gae(np.array([2.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
        # Advantages should be finite and non-zero
        assert np.all(np.isfinite(buf.advantages))
        assert np.any(buf.advantages != 0)

    def test_batch_generator_total_count(self):
        buf = RolloutBuffer(4, 8, (8, 31, 28), 4, 4)
        for _ in range(8):
            buf.insert(
                np.random.randn(4, 8, 31, 28).astype(np.float32),
                np.random.randn(4, 4).astype(np.float32),
                np.random.randint(0, 4, size=4),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.zeros(4, dtype=bool),
                np.ones((4, 4), dtype=bool),
            )
        buf.compute_gae(np.zeros(4, dtype=np.float32), 0.99, 0.95)
        total = sum(b["actions"].shape[0] for b in buf.batch_generator(16, torch.device("cpu")))
        assert total == 32  # 4 envs * 8 steps


class TestPPO:
    def test_select_action(self, ppo):
        obs_grid = np.random.randn(4, 8, 31, 28).astype(np.float32)
        obs_scalars = np.random.randn(4, 4).astype(np.float32)
        masks = np.ones((4, 4), dtype=bool)
        actions, log_probs, values = ppo.select_action(obs_grid, obs_scalars, masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert all(0 <= a < 4 for a in actions)

    def test_action_masking(self, ppo):
        obs_grid = np.random.randn(100, 8, 31, 28).astype(np.float32)
        obs_scalars = np.random.randn(100, 4).astype(np.float32)
        masks = np.zeros((100, 4), dtype=bool)
        masks[:, 0] = True  # only action 0 is legal
        actions, _, _ = ppo.select_action(obs_grid, obs_scalars, masks)
        assert np.all(actions == 0)

    def test_update_returns_finite_losses(self, ppo):
        buf = RolloutBuffer(4, 16, (8, 31, 28), 4, 4)
        for _ in range(16):
            buf.insert(
                np.random.randn(4, 8, 31, 28).astype(np.float32),
                np.random.randn(4, 4).astype(np.float32),
                np.random.randint(0, 4, size=4),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                np.zeros(4, dtype=bool),
                np.ones((4, 4), dtype=bool),
            )
        buf.compute_gae(np.zeros(4, dtype=np.float32), 0.99, 0.95)
        metrics = ppo.update(buf)
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])

    def test_lr_annealing(self, ppo):
        initial_lr = ppo.optimizer.param_groups[0]["lr"]
        ppo.anneal_lr(500, 1000)
        new_lr = ppo.optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr
        assert new_lr > 0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_ppo.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pacman/agents/ppo.py tests/test_ppo.py
git commit -m "feat(v2): PPO algorithm with GAE, clipped surrogate, action masking + tests"
```

---

### Task 12: Checkpoint Manager — `pacman/training/checkpoint.py`

**Files:**
- Create: `pacman/training/checkpoint.py`

- [ ] **Step 1: Write checkpoint save/load**

```python
# pacman/training/checkpoint.py
"""Checkpoint save/load for PPO training."""
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
    }
```

- [ ] **Step 2: Commit**

```bash
git add pacman/training/checkpoint.py
git commit -m "feat(v2): checkpoint save/load with best-model tracking"
```

---

### Task 13: Evaluator — `pacman/training/evaluator.py`

**Files:**
- Create: `pacman/training/evaluator.py`

- [ ] **Step 1: Write evaluator**

```python
# pacman/training/evaluator.py
"""Greedy policy evaluation."""
import numpy as np
import torch

from ..env.pacman_env import PacmanEnv
from ..agents.networks import ActorCritic


class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.env = PacmanEnv(config, difficulty=2)  # always eval at full difficulty

    @torch.no_grad()
    def evaluate(
        self,
        network: ActorCritic,
        num_episodes: int,
        device: torch.device,
    ) -> dict:
        network.eval()
        scores = []
        steps_list = []
        ghosts_eaten = []
        cleared = 0

        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=ep)
            episode_ghosts = 0
            while True:
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                mask = self.env.get_legal_mask()
                mask_t = torch.as_tensor(mask[None], device=device)
                logits, _ = network(grid_t, scalars_t, mask_t)
                action = logits.argmax(dim=-1).item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                for event in info.get("events", []):
                    if event.startswith("eat_ghost"):
                        episode_ghosts += 1
                if terminated or truncated:
                    scores.append(info["score"])
                    steps_list.append(self.env.state.step_count)
                    ghosts_eaten.append(episode_ghosts)
                    if info.get("winner") == "pacman":
                        cleared += 1
                    break

        network.train()
        return {
            "level_clear_rate": cleared / max(num_episodes, 1),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
            "mean_ghosts_eaten": float(np.mean(ghosts_eaten)) if ghosts_eaten else 0.0,
        }
```

- [ ] **Step 2: Commit**

```bash
git add pacman/training/evaluator.py
git commit -m "feat(v2): greedy policy evaluator"
```

---

### Task 14: Trainer — `pacman/training/trainer.py` + Integration Tests

**Files:**
- Create: `pacman/training/trainer.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write trainer**

```python
# pacman/training/trainer.py
"""PPO training orchestrator."""
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..env.vec_env import VecEnv
from ..agents.networks import ActorCritic
from ..agents.ppo import PPO
from ..agents.rollout import RolloutBuffer
from ..training.evaluator import Evaluator
from ..training.checkpoint import save_checkpoint, load_checkpoint


class RunningMeanStd:
    """Tracks running mean/std for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        total = self.count + count
        new_mean = self.mean + delta * count / total
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + delta ** 2 * self.count * count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self._clip, self._clip,
        )

    @property
    def _clip(self):
        return 10.0

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


def get_device(config: dict) -> torch.device:
    choice = config.get("device", "auto")
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(choice)


class Trainer:
    def __init__(self, config: dict, run_dir: Path, resume: bool = False):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(config)

        # Environment
        env_cfg = config["env"]
        self.vec_env = VecEnv(env_cfg["num_envs"], config, difficulty=0)

        # Network
        net_cfg = config["network"]
        self.network = ActorCritic(
            cnn_channels=net_cfg["cnn_channels"],
            cnn_kernels=net_cfg["cnn_kernels"],
            cnn_strides=net_cfg["cnn_strides"],
            shared_hidden=net_cfg["shared_hidden"],
            head_hidden=net_cfg["head_hidden"],
        )

        # PPO
        self.ppo = PPO(self.network, config, self.device)

        # Rollout buffer
        self.rollout = RolloutBuffer(
            num_envs=env_cfg["num_envs"],
            rollout_steps=config["ppo"]["rollout_steps"],
            grid_shape=(env_cfg["observation_channels"], 31, 28),
            num_scalars=env_cfg["num_scalar_features"],
            num_actions=4,
        )

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # Evaluator
        self.evaluator = Evaluator(config)

        # Logging
        self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))

        # Curriculum
        self.curriculum_phase = 0
        self.best_clear_rate = 0.0
        self.start_update = 0

        if resume:
            ckpt_dir = self.run_dir / "checkpoints"
            if (ckpt_dir / "latest.pt").exists():
                meta = load_checkpoint(ckpt_dir, self.network, self.ppo.optimizer)
                self.start_update = meta["update"] + 1
                self.curriculum_phase = meta.get("curriculum_phase", 0)
                norm_state = meta.get("reward_normalizer", {})
                if norm_state:
                    self.reward_normalizer.load_state_dict(norm_state)

    def train(self, total_updates: int | None = None) -> None:
        total = total_updates or self.config["training"]["total_updates"]
        train_cfg = self.config["training"]
        ppo_cfg = self.config["ppo"]
        curriculum_cfg = self.config["curriculum"]
        N = self.config["env"]["num_envs"]
        T = ppo_cfg["rollout_steps"]

        obs = self.vec_env.reset(seed=42)

        for update in range(self.start_update, total):
            t_start = time.time()

            # --- Anneal schedules ---
            self.ppo.anneal_lr(update, total)
            entropy_coef = self._get_entropy_coef(update, total)
            self.ppo.set_entropy_coef(entropy_coef)

            # --- Advance curriculum ---
            self._advance_curriculum(update, curriculum_cfg)

            # --- Collect rollout ---
            self.rollout.reset()
            for _step in range(T):
                legal_masks = self.vec_env.get_legal_masks()
                actions, log_probs, values = self.ppo.select_action(
                    obs["grid"], obs["scalars"], legal_masks,
                )
                next_obs, rewards, dones, infos = self.vec_env.step(actions)

                # Normalize rewards
                self.reward_normalizer.update(rewards)
                norm_rewards = self.reward_normalizer.normalize(rewards)

                self.rollout.insert(
                    obs["grid"], obs["scalars"],
                    actions, log_probs, values,
                    norm_rewards, dones, legal_masks,
                )
                obs = next_obs

            # --- Compute GAE ---
            legal_masks = self.vec_env.get_legal_masks()
            last_values = self.ppo.get_value(obs["grid"], obs["scalars"], legal_masks)
            self.rollout.compute_gae(
                last_values, ppo_cfg["gamma"], ppo_cfg["gae_lambda"],
            )

            # --- PPO Update ---
            metrics = self.ppo.update(self.rollout)

            # --- Logging ---
            fps = N * T / (time.time() - t_start)
            if update % train_cfg["log_every"] == 0:
                self.writer.add_scalar("loss/policy", metrics["policy_loss"], update)
                self.writer.add_scalar("loss/value", metrics["value_loss"], update)
                self.writer.add_scalar("loss/entropy", metrics["entropy"], update)
                self.writer.add_scalar("schedule/learning_rate",
                                       self.ppo.optimizer.param_groups[0]["lr"], update)
                self.writer.add_scalar("schedule/entropy_coef", entropy_coef, update)
                self.writer.add_scalar("schedule/curriculum_phase",
                                       self.curriculum_phase, update)
                self.writer.add_scalar("throughput/fps", fps, update)

            # --- Evaluation ---
            if update % train_cfg["eval_every"] == 0 and update > 0:
                eval_result = self.evaluator.evaluate(
                    self.network, train_cfg["eval_episodes"], self.device,
                )
                self.writer.add_scalar("performance/level_clear_rate",
                                       eval_result["level_clear_rate"], update)
                self.writer.add_scalar("performance/mean_score",
                                       eval_result["mean_score"], update)
                self.writer.add_scalar("performance/mean_steps",
                                       eval_result["mean_steps"], update)
                is_best = eval_result["level_clear_rate"] > self.best_clear_rate
                if is_best:
                    self.best_clear_rate = eval_result["level_clear_rate"]
                print(f"[Update {update}] clear={eval_result['level_clear_rate']:.1%} "
                      f"score={eval_result['mean_score']:.0f} fps={fps:.0f}")

            # --- Checkpoint ---
            if update % train_cfg["checkpoint_every"] == 0 and update > 0:
                is_best = False  # already handled above
                save_checkpoint(
                    self.run_dir / "checkpoints", update,
                    self.network, self.ppo.optimizer,
                    self.reward_normalizer.state_dict(),
                    self.curriculum_phase, self.config,
                    is_best=False,
                )

        # Final checkpoint
        save_checkpoint(
            self.run_dir / "checkpoints", total - 1,
            self.network, self.ppo.optimizer,
            self.reward_normalizer.state_dict(),
            self.curriculum_phase, self.config,
            is_best=False,
        )
        self.writer.close()

    def _get_entropy_coef(self, update: int, total: int) -> float:
        cfg = self.config["ppo"]
        start = cfg["entropy_coef_start"]
        end = cfg["entropy_coef_end"]
        anneal_frac = cfg["entropy_anneal_fraction"]
        if update < total * anneal_frac:
            return start
        progress = (update - total * anneal_frac) / (total * (1 - anneal_frac))
        return start + (end - start) * min(progress, 1.0)

    def _advance_curriculum(self, update: int, curriculum_cfg: dict) -> None:
        thresholds = curriculum_cfg["phase_thresholds"]
        difficulties = curriculum_cfg["difficulties"]
        for phase, threshold in enumerate(thresholds):
            if update >= threshold:
                if self.curriculum_phase != phase:
                    self.curriculum_phase = phase
                    self.vec_env.set_difficulty(difficulties[phase])
                    print(f"[Update {update}] Curriculum → phase {phase} "
                          f"(difficulty={difficulties[phase]})")
```

- [ ] **Step 2: Write integration tests**

```python
# tests/test_training.py
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.agents.ppo import PPO
from pacman.training.trainer import Trainer, get_device
from pacman.training.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def config():
    c = load_config()
    # Override for fast tests
    c["env"]["num_envs"] = 4
    c["ppo"]["rollout_steps"] = 16
    c["ppo"]["minibatch_size"] = 16
    c["ppo"]["num_epochs"] = 2
    c["training"]["total_updates"] = 5
    c["training"]["eval_every"] = 3
    c["training"]["eval_episodes"] = 2
    c["training"]["checkpoint_every"] = 2
    c["game"]["max_steps"] = 200
    return c


class TestTrainer:
    def test_short_training_completes(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            # Should complete without errors
            assert (Path(tmpdir) / "checkpoints" / "latest.pt").exists()

    def test_checkpoint_save_load(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=3)

            # Load checkpoint into fresh network
            network2 = ActorCritic()
            meta = load_checkpoint(Path(tmpdir) / "checkpoints", network2)
            assert meta["update"] >= 0

    def test_curriculum_advances(self, config):
        config["curriculum"]["phase_thresholds"] = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            assert trainer.curriculum_phase >= 1


class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            network = ActorCritic()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            save_checkpoint(
                Path(tmpdir), 100, network, optimizer,
                {"mean": 1.0, "var": 2.0, "count": 100},
                curriculum_phase=1, config={"test": True},
            )
            network2 = ActorCritic()
            optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-3)
            meta = load_checkpoint(Path(tmpdir), network2, optimizer2)
            assert meta["update"] == 100
            assert meta["curriculum_phase"] == 1
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_training.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add pacman/training/trainer.py tests/test_training.py
git commit -m "feat(v2): PPO trainer with curriculum, reward normalization, TensorBoard + tests"
```

---

### Task 15: Sprites — `pacman/viz/sprites.py`

**Files:**
- Create: `pacman/viz/sprites.py` (carry forward from v1 `src/gui/sprites.py`)

- [ ] **Step 1: Copy and adapt sprites from v1**

Copy `src/gui/sprites.py` to `pacman/viz/sprites.py`. Update the import to use `pacman.engine.constants` instead of `src.engine.constants`. No functional changes — the sprite drawing functions (`draw_pacman`, `draw_ghost`, `draw_pellet`, `draw_power_pellet`, `draw_fruit`) are reusable as-is.

- [ ] **Step 2: Commit**

```bash
git add pacman/viz/sprites.py
git commit -m "feat(v2): carry forward sprite drawing from v1"
```

---

### Task 16: Renderer — `pacman/viz/renderer.py`

**Files:**
- Create: `pacman/viz/renderer.py`

- [ ] **Step 1: Write Pygame renderer adapted for v2 GameState**

Adapt `src/gui/renderer.py` to accept the v2 `GameState` dataclass. Key changes:
- Remove per-ghost agent stats from sidebar
- Add action probability bar chart and V(s) gauge
- Add keyboard controls: SPACE=pause, N=step, R=reset, Q=quit
- Accept optional `agent_info` dict with `action_probs` and `value` for sidebar display

This is a significant file — adapt the v1 structure with the new sidebar layout per spec section 11.2.

- [ ] **Step 2: Commit**

```bash
git add pacman/viz/renderer.py
git commit -m "feat(v2): Pygame renderer with action probs sidebar and keyboard controls"
```

---

### Task 17: Dashboard — `pacman/viz/dashboard.py`

**Files:**
- Create: `pacman/viz/dashboard.py`

- [ ] **Step 1: Write TensorBoard-reading dashboard**

Adapt `src/gui/dashboard.py` to read from TensorBoard event files instead of SQLite. Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` to read scalar data. Charts per spec section 11.3: score progression, level clear rate, loss curves, episode length, ghost kill stats.

- [ ] **Step 2: Commit**

```bash
git add pacman/viz/dashboard.py
git commit -m "feat(v2): Dash training dashboard reading TensorBoard logs"
```

---

### Task 18: Scripts — `scripts/train.py`, `scripts/evaluate.py`, `scripts/watch.py`

**Files:**
- Overwrite: `scripts/train.py`, `scripts/evaluate.py`, `scripts/watch.py`

- [ ] **Step 1: Write train.py**

```python
# scripts/train.py
"""Train Pac-Man PPO agent."""
import argparse
from datetime import datetime
from pathlib import Path

from pacman.utils.config import load_config
from pacman.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Pac-Man PPO agent")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--run-dir", type=str, default=None, help="Run output directory")
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_envs:
        config["env"]["num_envs"] = args.num_envs
    if args.device:
        config["device"] = args.device

    run_dir = args.run_dir or f"runs/{datetime.now():%Y-%m-%d_%H-%M-%S}"
    trainer = Trainer(config, Path(run_dir), resume=args.resume)

    try:
        trainer.train(total_updates=args.total_updates)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        from pacman.training.checkpoint import save_checkpoint
        save_checkpoint(
            Path(run_dir) / "checkpoints",
            trainer.start_update, trainer.network, trainer.ppo.optimizer,
            trainer.reward_normalizer.state_dict(),
            trainer.curriculum_phase, config,
        )
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write evaluate.py**

```python
# scripts/evaluate.py
"""Evaluate a trained Pac-Man agent from checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.training.checkpoint import load_checkpoint
from pacman.training.evaluator import Evaluator
from pacman.training.trainer import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to training run directory")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)
    network = ActorCritic()
    ckpt_path = Path(args.run_dir) / "checkpoints"
    meta = load_checkpoint(ckpt_path, network, filename=args.checkpoint)
    print(f"Loaded checkpoint: update={meta['update']}")

    evaluator = Evaluator(config)
    results = evaluator.evaluate(network, args.episodes, device)

    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Level clear rate: {results['level_clear_rate']:.1%}")
    print(f"  Mean score:       {results['mean_score']:.0f}")
    print(f"  Mean steps:       {results['mean_steps']:.0f}")
    print(f"  Mean ghosts eaten:{results['mean_ghosts_eaten']:.1f}")

    out_path = Path(args.run_dir) / "eval" / f"eval_{meta['update']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write watch.py**

```python
# scripts/watch.py
"""Watch a trained Pac-Man agent play in Pygame."""
import argparse
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.env.pacman_env import PacmanEnv
from pacman.training.checkpoint import load_checkpoint
from pacman.training.trainer import get_device
from pacman.viz.renderer import GameRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)

    network = None
    if args.run_dir:
        network = ActorCritic()
        load_checkpoint(Path(args.run_dir) / "checkpoints", network, filename=args.checkpoint)
        network.to(device).eval()

    env = PacmanEnv(config, difficulty=2)
    renderer = GameRenderer(config)
    obs, _ = env.reset(seed=args.seed)
    paused = False

    while True:
        action_probs = None
        value = None

        if network is not None:
            with torch.no_grad():
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                mask = env.get_legal_mask()
                mask_t = torch.as_tensor(mask[None], device=device)
                logits, val = network(grid_t, scalars_t, mask_t)
                probs = torch.softmax(logits, dim=-1)
                action_probs = probs[0].cpu().numpy()
                value = val[0].item()
                action = logits.argmax(dim=-1).item()
        else:
            mask = env.get_legal_mask()
            action = np.random.choice(np.where(mask)[0])

        agent_info = {"action_probs": action_probs, "value": value}
        running, step_requested = renderer.render(env.state, agent_info)
        if not running:
            break

        if not paused or step_requested:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, _ = env.reset()

        renderer.tick(args.fps)

    renderer.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py scripts/evaluate.py scripts/watch.py
git commit -m "feat(v2): CLI scripts — train, evaluate, watch"
```

---

### Task 19: End-to-End Verification

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass.

- [ ] **Step 2: Smoke-test training (10 updates)**

Run: `python scripts/train.py --total-updates 10 --num-envs 8 --device cpu`
Expected: Completes, prints evaluation metrics, creates `runs/` directory with checkpoints and tensorboard logs.

- [ ] **Step 3: Verify TensorBoard**

Run: `tensorboard --logdir runs/`
Expected: Dashboard loads with loss curves, schedule plots.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(v2): end-to-end verification complete — all tests passing, training pipeline operational"
```
