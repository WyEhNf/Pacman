# pacman/env/vec_env.py
"""Vectorized Pac-Man environment — N games as batched NumPy operations."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, GhostID, Direction, DIRECTION_DELTAS, OPPOSITE_DIRECTION,
    MAZE_ROWS, MAZE_COLS, NUM_GHOSTS, NUM_ACTIONS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths, is_walkable
from ..engine.maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, FRUIT_POSITION,
)
from ..engine.entities import create_initial_state
from ..engine.game import step_game, get_legal_actions
from ..engine import ghost_ai
from .pacman_env import NUM_CHANNELS, NUM_SCALARS


class VecEnv:
    """Vectorized Pac-Man environment stepping N games in parallel.

    Uses per-game step_game() internally with auto-reset.
    Supports frame stacking for temporal observations.
    """

    def __init__(self, num_envs: int, config: dict, difficulty: int = 0):
        self.num_envs = num_envs
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._states = []
        self._rngs = []

        # Frame stacking
        self.frame_stack = config["env"].get("frame_stack", 1)
        self._frame_buffer = None  # (N, frame_stack, C, H, W)

    def reset(self, seed: int | None = None) -> dict:
        base_seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._states = []
        self._rngs = []
        for i in range(self.num_envs):
            self._states.append(create_initial_state(self.config, self.difficulty))
            self._rngs.append(np.random.default_rng(base_seed + i))

        raw_obs = self._build_batch_obs()

        if self.frame_stack > 1:
            # Fill all frame slots with the initial observation
            self._frame_buffer = np.tile(
                raw_obs["grid"][:, np.newaxis],  # (N, 1, C, H, W)
                (1, self.frame_stack, 1, 1, 1),
            )

        return self._stack_obs(raw_obs)

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

        raw_obs = self._build_batch_obs()

        # Update frame buffer
        if self.frame_stack > 1:
            self._frame_buffer[:, :-1] = self._frame_buffer[:, 1:]
            self._frame_buffer[:, -1] = raw_obs["grid"]
            # Reset frame buffer for done environments (new episode = no history)
            for i in range(self.num_envs):
                if dones[i]:
                    self._frame_buffer[i, :] = raw_obs["grid"][i]

        obs = self._stack_obs(raw_obs)
        return obs, rewards, dones, infos

    def get_legal_masks(self) -> np.ndarray:
        """Return (N, 4) bool mask of legal actions per env."""
        masks = np.zeros((self.num_envs, NUM_ACTIONS), dtype=bool)
        for i in range(self.num_envs):
            masks[i] = get_legal_actions(
                self._states[i].grid, self._states[i].pac_pos,
                prev_dir=int(self._states[i].pac_dir),
            )
        return masks

    def set_difficulty(self, difficulty: int) -> None:
        self.difficulty = difficulty
        for state in self._states:
            state.difficulty = difficulty

    def _stack_obs(self, raw_obs: dict) -> dict:
        """Stack frames if frame_stack > 1, otherwise return raw obs."""
        if self.frame_stack <= 1:
            return raw_obs
        stacked = self._frame_buffer.reshape(
            self.num_envs, -1, MAZE_ROWS, MAZE_COLS,
        )
        return {"grid": stacked.copy(), "scalars": raw_obs["scalars"]}

    def _build_batch_obs(self) -> dict:
        """Build raw (unstacked) batched observations: grid (N,14,31,28), scalars (N,15)."""
        grids = np.zeros((self.num_envs, NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)
        scalars = np.zeros((self.num_envs, NUM_SCALARS), dtype=np.float32)
        max_fright = self.config["game"]["frightened_duration"]
        max_lives = self.config["game"]["lives"]

        for i, s in enumerate(self._states):
            pr, pc = int(s.pac_pos[0]), int(s.pac_pos[1])

            # Ch 0: Walls
            grids[i, 0] = (s.grid == Tile.WALL)
            # Ch 1: Pac-Man position
            grids[i, 1, pr, pc] = 1.0
            # Ch 2: Pellets
            grids[i, 2] = (s.grid == Tile.PELLET)
            # Ch 3: Power pellets
            grids[i, 3] = (s.grid == Tile.POWER_PELLET)

            # Ch 4-7: Ghost target heatmaps
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
                            grids[i, 4 + g, rr, cc] = max(grids[i, 4 + g, rr, cc], val)

            # Ch 8: Edible ghosts (frightened)
            for g in range(NUM_GHOSTS):
                if not s.ghost_in_house[g] and s.ghost_mode[g] == GhostMode.FRIGHTENED:
                    grids[i, 8, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0

            # Ch 9: Eaten ghosts (returning to house)
            for g in range(NUM_GHOSTS):
                if not s.ghost_in_house[g] and s.ghost_mode[g] == GhostMode.EATEN:
                    grids[i, 9, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0

            # Ch 10: Ghost house + door
            grids[i, 10] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR))

            # Ch 11: Fruit
            if s.fruit_active:
                grids[i, 11, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

            # Ch 12: Dangerous ghosts (scatter/chase)
            for g in range(NUM_GHOSTS):
                if not s.ghost_in_house[g] and s.ghost_mode[g] in (GhostMode.SCATTER, GhostMode.CHASE):
                    grids[i, 12, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0

            # Ch 13: Pac-Man direction arrow (2-cell indicator)
            dr, dc = DIRECTION_DELTAS[int(s.pac_dir)]
            grids[i, 13, pr, pc] = 1.0
            nr, nc = pr + dr, (pc + dc) % MAZE_COLS
            if is_walkable(s.grid, nr, nc, for_ghost=False):
                grids[i, 13, nr, nc] = 0.6

            # Ch 14-17: Ghost next-step positions (lightweight MCTS lookahead)
            for g in range(NUM_GHOSTS):
                if s.ghost_in_house[g] or s.ghost_mode[g] == GhostMode.EATEN:
                    continue
                gr_g, gc_g = int(s.ghost_pos[g, 0]), int(s.ghost_pos[g, 1])
                if s.ghost_mode[g] == GhostMode.FRIGHTENED:
                    grids[i, 14 + g, gr_g, gc_g] = 1.0
                    continue
                target = ghost_ai.compute_ghost_target(
                    g, int(s.ghost_mode[g]), s.pac_pos, int(s.pac_dir),
                    s.ghost_pos, s.difficulty, s.pellets_remaining,
                )
                next_dir = ghost_ai.choose_direction_toward_target(
                    s.grid, gr_g, gc_g, int(s.ghost_dir[g]),
                    target[0], target[1],
                )
                ndr, ndc = DIRECTION_DELTAS[next_dir]
                nr_g, nc_g = gr_g + ndr, (gc_g + ndc) % MAZE_COLS
                if is_walkable(s.grid, nr_g, nc_g, for_ghost=True):
                    grids[i, 14 + g, nr_g, nc_g] = 1.0

            # Scalars (15)
            active = 0
            dists, dirs = [], []
            for g in range(NUM_GHOSTS):
                if s.ghost_in_house[g] or s.ghost_mode[g] == GhostMode.EATEN:
                    dists.append(40.0)
                    dirs.append(-1.0)
                else:
                    active += 1
                    dist = abs(int(s.ghost_pos[g, 0]) - pr) + abs(int(s.ghost_pos[g, 1]) - pc)
                    dists.append(min(float(dist), 40.0))
                    dirs.append(float(s.ghost_dir[g]))

            scalars[i] = [
                s.pac_power_timer / max(max_fright, 1),    # 0
                s.pac_lives / max_lives,                    # 1
                s.pac_ghosts_eaten / 4.0,                   # 2
                s.pellets_eaten / max(s.total_pellets, 1),  # 3
                s.pac_dir / 3.0,                            # 4
                1.0 if s.pac_powered else 0.0,              # 5
                active / 4.0,                               # 6
                dists[0] / 40.0, dists[1] / 40.0,          # 7-8
                dists[2] / 40.0, dists[3] / 40.0,          # 9-10
                dirs[0] / 3.0, dirs[1] / 3.0,              # 11-12
                dirs[2] / 3.0, dirs[3] / 3.0,              # 13-14
            ]

        return {"grid": grids, "scalars": scalars}
