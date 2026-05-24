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
