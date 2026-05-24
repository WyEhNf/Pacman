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


def get_legal_actions(
    grid: np.ndarray, pac_pos: np.ndarray, prev_dir: int = -1,
) -> np.ndarray:
    """Return (4,) bool mask of legal Pac-Man moves.

    If prev_dir is provided, the reverse direction is masked out (arcade-style
    no-reverse constraint). This eliminates oscillation by making it physically
    impossible to go backward — unless reversing is the only legal move.
    """
    row, col = int(pac_pos[0]), int(pac_pos[1])
    mask = np.zeros(4, dtype=bool)
    for d in Direction:
        dr, dc = DIRECTION_DELTAS[d]
        nr, nc = row + dr, (col + dc) % MAZE_COLS
        mask[d] = is_walkable(grid, nr, nc, for_ghost=False)

    # Mask the reverse direction unless it's the only legal move
    if prev_dir >= 0:
        reverse = OPPOSITE_DIRECTION[Direction(prev_dir)]
        if mask.sum() > 1 and mask[reverse]:
            mask[reverse] = False

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
        if state.pac_lives == config["game"]["lives"]:
            events.append("clear_perfect")
        else:
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
    """Check if ghosts should exit the house based on pellets eaten OR time.

    Ghosts exit when EITHER condition is met (whichever comes first):
    - Enough pellets have been eaten (pellet threshold)
    - Enough game steps have passed (timer threshold)
    This ensures ghosts always enter the game even if the agent isn't eating.
    """
    pellet_thresholds = config["game"]["ghost_exit_pellets"]
    timer_thresholds = config["game"].get("ghost_exit_timer", [0, 0, 0, 0])
    for i in range(NUM_GHOSTS):
        if state.ghost_in_house[i] and not state.ghost_exiting[i]:
            by_pellets = state.pellets_eaten >= pellet_thresholds[i]
            by_timer = state.step_count >= timer_thresholds[i]
            if by_pellets or by_timer:
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
        elif event == "clear_perfect":
            reward += reward_cfg.get("clear_perfect", reward_cfg.get("clear_level", 50.0))
        elif event == "death":
            reward += reward_cfg["death"]
        elif event == "game_over":
            reward += reward_cfg["game_over"]

    # Ghost proximity penalty — teaches avoidance even in scatter phase
    proximity_penalty = reward_cfg.get("ghost_proximity", 0.0)
    if proximity_penalty != 0.0:
        pr, pc = int(state.pac_pos[0]), int(state.pac_pos[1])
        for i in range(NUM_GHOSTS):
            if state.ghost_in_house[i]:
                continue
            if state.ghost_mode[i] in (GhostMode.FRIGHTENED, GhostMode.EATEN):
                continue
            gr, gc = int(state.ghost_pos[i, 0]), int(state.ghost_pos[i, 1])
            dist = abs(gr - pr) + abs(gc - pc)
            if dist <= 2:
                reward += proximity_penalty

    # Edible ghost proximity bonus — encourages chasing frightened ghosts
    edible_bonus = reward_cfg.get("edible_proximity", 0.0)
    if edible_bonus > 0.0 and state.pac_powered:
        pr, pc = int(state.pac_pos[0]), int(state.pac_pos[1])
        for i in range(NUM_GHOSTS):
            if state.ghost_in_house[i]:
                continue
            if state.ghost_mode[i] != GhostMode.FRIGHTENED:
                continue
            gr, gc = int(state.ghost_pos[i, 0]), int(state.ghost_pos[i, 1])
            dist = abs(gr - pr) + abs(gc - pc)
            if dist <= 6:
                reward += edible_bonus * (7 - dist)

    return reward
