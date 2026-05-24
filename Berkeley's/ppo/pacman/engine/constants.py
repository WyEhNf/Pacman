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
