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
