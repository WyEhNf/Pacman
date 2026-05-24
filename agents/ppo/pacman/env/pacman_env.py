# pacman/env/pacman_env.py
"""Single Pac-Man environment with Gymnasium-compatible interface."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, DIRECTION_DELTAS, MAZE_ROWS, MAZE_COLS, NUM_GHOSTS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths, is_walkable
from ..engine.maze_data import FRUIT_POSITION
from ..engine.entities import create_initial_state, GameState
from ..engine.game import step_game, get_legal_actions
from ..engine import ghost_ai

NUM_CHANNELS = 18
NUM_SCALARS = 15


class PacmanEnv:
    """Single Pac-Man environment for evaluation and visualization."""

    def __init__(self, config: dict, difficulty: int = 0):
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._state: GameState | None = None
        self._rng = np.random.default_rng()

        # Frame stacking
        self.frame_stack = config["env"].get("frame_stack", 1)
        self._frame_buffer = None  # (frame_stack, C, H, W)

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = create_initial_state(self.config, self.difficulty)
        raw_obs = self._build_obs()

        if self.frame_stack > 1:
            self._frame_buffer = np.tile(
                raw_obs["grid"][np.newaxis],  # (1, C, H, W)
                (self.frame_stack, 1, 1, 1),
            )

        return self._stack_obs(raw_obs), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        state, events, reward = step_game(
            self._state, action, self.config, self._return_paths, self._rng,
        )
        self._state = state
        raw_obs = self._build_obs()

        if self.frame_stack > 1:
            self._frame_buffer[:-1] = self._frame_buffer[1:]
            self._frame_buffer[-1] = raw_obs["grid"]

        obs = self._stack_obs(raw_obs)
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
        return get_legal_actions(
            self._state.grid, self._state.pac_pos,
            prev_dir=int(self._state.pac_dir),
        )

    @property
    def state(self) -> GameState:
        return self._state

    def _stack_obs(self, raw_obs: dict) -> dict:
        """Stack frames if frame_stack > 1, otherwise return raw obs."""
        if self.frame_stack <= 1:
            return raw_obs
        stacked = self._frame_buffer.reshape(-1, MAZE_ROWS, MAZE_COLS)
        return {"grid": stacked.copy(), "scalars": raw_obs["scalars"]}

    def _build_obs(self) -> dict:
        """Build raw (unstacked) 14-channel grid + 15 scalars observation."""
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

        # Ch 4-7: Ghost target heatmaps (Blinky/Pinky/Inky/Clyde)
        for g in range(NUM_GHOSTS):
            if s.ghost_in_house[g]:
                continue
            if s.ghost_mode[g] in (GhostMode.EATEN, GhostMode.FRIGHTENED):
                continue
            target = ghost_ai.compute_ghost_target(
                g, int(s.ghost_mode[g]), s.pac_pos, int(s.pac_dir),
                s.ghost_pos, s.difficulty, s.pellets_remaining,
            )
            tr, tc = target
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = tr + dr
                    cc = (tc + dc) % MAZE_COLS
                    if 0 <= rr < MAZE_ROWS:
                        val = 1.0 if dr == 0 and dc == 0 else 0.5
                        grid[4 + g, rr, cc] = max(grid[4 + g, rr, cc], val)

        # Ch 8: Edible ghosts (frightened)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] == GhostMode.FRIGHTENED:
                grid[8, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0

        # Ch 9: Eaten ghosts (returning to house) — previously invisible
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] == GhostMode.EATEN:
                grid[9, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0

        # Ch 10: Ghost house + door
        grid[10] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR)).astype(np.float32)

        # Ch 11: Fruit
        if s.fruit_active:
            grid[11, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

        # Ch 12: Dangerous ghosts (scatter/chase)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] in (GhostMode.SCATTER, GhostMode.CHASE):
                grid[12, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0

        # Ch 13: Pac-Man direction arrow (2-cell indicator)
        dr, dc = DIRECTION_DELTAS[int(s.pac_dir)]
        pr, pc = int(s.pac_pos[0]), int(s.pac_pos[1])
        grid[13, pr, pc] = 1.0
        nr, nc = pr + dr, (pc + dc) % MAZE_COLS
        if is_walkable(s.grid, nr, nc, for_ghost=False):
            grid[13, nr, nc] = 0.6

        # Ch 14-17: Ghost next-step positions (lightweight MCTS lookahead)
        for g in range(NUM_GHOSTS):
            if s.ghost_in_house[g] or s.ghost_mode[g] == GhostMode.EATEN:
                continue
            gr, gc = int(s.ghost_pos[g, 0]), int(s.ghost_pos[g, 1])
            if s.ghost_mode[g] == GhostMode.FRIGHTENED:
                # Frightened: move is random, show current position as best guess
                grid[14 + g, gr, gc] = 1.0
                continue
            # Deterministic ghost: compute target and next direction
            target = ghost_ai.compute_ghost_target(
                g, int(s.ghost_mode[g]), s.pac_pos, int(s.pac_dir),
                s.ghost_pos, s.difficulty, s.pellets_remaining,
            )
            next_dir = ghost_ai.choose_direction_toward_target(
                s.grid, gr, gc, int(s.ghost_dir[g]),
                target[0], target[1],
            )
            ndr, ndc = DIRECTION_DELTAS[next_dir]
            nr_ghost, nc_ghost = gr + ndr, (gc + ndc) % MAZE_COLS
            if is_walkable(s.grid, nr_ghost, nc_ghost, for_ghost=True):
                grid[14 + g, nr_ghost, nc_ghost] = 1.0

        # Scalars (15)
        max_fright = self.config["game"]["frightened_duration"]
        max_lives = self.config["game"]["lives"]
        active_count = 0
        distances = []
        directions = []
        for i in range(NUM_GHOSTS):
            if s.ghost_in_house[i] or s.ghost_mode[i] == GhostMode.EATEN:
                distances.append(40.0)
                directions.append(-1.0)
            else:
                active_count += 1
                dist = abs(int(s.ghost_pos[i, 0]) - pr) + abs(int(s.ghost_pos[i, 1]) - pc)
                distances.append(min(float(dist), 40.0))
                directions.append(float(s.ghost_dir[i]))

        scalars = np.array([
            s.pac_power_timer / max(max_fright, 1),       # 0
            s.pac_lives / max_lives,                       # 1
            s.pac_ghosts_eaten / 4.0,                      # 2
            s.pellets_eaten / max(s.total_pellets, 1),     # 3
            s.pac_dir / 3.0,                               # 4
            1.0 if s.pac_powered else 0.0,                 # 5: powered_binary
            active_count / 4.0,                            # 6: active_ghosts
            distances[0] / 40.0,                           # 7: dist_blinky
            distances[1] / 40.0,                           # 8: dist_pinky
            distances[2] / 40.0,                           # 9: dist_inky
            distances[3] / 40.0,                           # 10: dist_clyde
            directions[0] / 3.0,                           # 11: blinky_dir
            directions[1] / 3.0,                           # 12: pinky_dir
            directions[2] / 3.0,                           # 13: inky_dir
            directions[3] / 3.0,                           # 14: clyde_dir
        ], dtype=np.float32)

        return {"grid": grid, "scalars": scalars}
