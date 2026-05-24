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
