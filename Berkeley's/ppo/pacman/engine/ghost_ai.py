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
