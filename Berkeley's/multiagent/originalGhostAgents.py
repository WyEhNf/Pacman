"""Original 1980 Pac-Man ghost AI for Berkeley skeleton.

Ghost IDs: 0=Blinky(Pursuit), 1=Pinky(Ambush), 2=Inky(Flank), 3=Clyde(Fickle)
Scatter targets are actual corner positions on the classic 28x31 maze.
"""
import numpy as np
from game import Agent, Directions

DIR_ORDER = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]
DIR_DELTA = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
             Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}
OPPOSITE = {Directions.NORTH: Directions.SOUTH, Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST, Directions.WEST: Directions.EAST}

# Mode schedule: (game_ticks, mode) alternating scatter/chase, -1 = forever
# Scaled for Berkeley: each tick ~1 step
MODE_SCHEDULE = [(84, 0), (240, 1), (84, 0), (240, 1), (60, 0), (240, 1), (60, 0), (-1, 1)]

# Scatter targets: (x, y) — actual corners of the 28x31 classic maze
# Blinky: top-right, Pinky: top-left, Inky: bottom-right, Clyde: bottom-left
SCATTER_TARGETS = {
    0: (25, 1),     # Blinky: near top-right corner
    1: (1, 1),      # Pinky: near top-left corner
    2: (26, 29),    # Inky: near bottom-right
    3: (1, 29),     # Clyde: near bottom-left
}

# Ghost house exit position (x, y)
GHOST_HOUSE_EXIT = (13, 13)  # center-top opening


class OriginalGhost(Agent):
    """Arcade-accurate ghost with personality-based targeting."""
    def __init__(self, index, ghost_id=0):
        super().__init__(index)
        self.ghost_id = ghost_id
        self.mode_timer = 0
        self.schedule_idx = 0
        self.current_mode = 0  # 0=scatter, 1=chase, 2=frightened
        self.last_pac_pos = None
        self.pac_dir = Directions.NORTH

    def _update_pac_dir(self, gameState):
        """Track Pac-Man direction by comparing positions."""
        pac_pos = gameState.getPacmanPosition()
        if self.last_pac_pos is not None:
            dx = pac_pos[0] - self.last_pac_pos[0]
            dy = pac_pos[1] - self.last_pac_pos[1]
            if dx != 0 or dy != 0:
                for d, (ddx, ddy) in DIR_DELTA.items():
                    if ddx == dx and ddy == dy:
                        self.pac_dir = d
                        break
        self.last_pac_pos = pac_pos

    def _update_mode(self, gameState):
        """Advance mode timer and handle mode transitions."""
        ghost_state = gameState.getGhostStates()[self.ghost_id]

        # Frightened override
        if ghost_state.scaredTimer > 0:
            self.current_mode = 2
            return 2

        # Resume normal mode schedule (don't advance timer during frightened)
        self.mode_timer -= 1
        if self.mode_timer <= 0 and self.schedule_idx < len(MODE_SCHEDULE):
            ticks, mode = MODE_SCHEDULE[self.schedule_idx]
            self.schedule_idx += 1
            self.mode_timer = ticks if ticks > 0 else 999999
            self.current_mode = mode
        return self.current_mode

    def _curr_dir(self, gameState):
        """Get ghost's current facing direction from Berkeley state."""
        gs = gameState.getGhostStates()[self.ghost_id]
        return gs.configuration.direction

    def _target(self, gameState, mode):
        """Compute target (x, y) based on ghost personality and mode."""
        pac_x, pac_y = gameState.getPacmanPosition()
        pac_x, pac_y = int(pac_x), int(pac_y)
        ghost_states = gameState.getGhostStates()
        W = gameState.getWalls().width
        H = gameState.getWalls().height

        if mode == 0:  # Scatter
            return SCATTER_TARGETS.get(self.ghost_id, (1, 1))

        elif mode == 1:  # Chase
            if self.ghost_id == 0:  # Blinky: straight pursuit
                return pac_x, pac_y

            elif self.ghost_id == 1:  # Pinky: 4 tiles ahead of Pac-Man
                dx, dy = DIR_DELTA.get(self.pac_dir, (0, 1))
                tx = pac_x + dx * 4
                ty = pac_y + dy * 4
                # Clamp to maze boundaries
                tx = max(1, min(W - 2, tx))
                ty = max(1, min(H - 2, ty))
                return tx, ty

            elif self.ghost_id == 2:  # Inky: double vector from Blinky
                # Blinky is ghost index 0 (first ghost in list)
                blinky = ghost_states[0] if len(ghost_states) > 0 else None
                bx, by = (int(blinky.getPosition()[0]), int(blinky.getPosition()[1])) if blinky else (13, 13)
                dx, dy = DIR_DELTA.get(self.pac_dir, (0, 1))
                ahead_x = pac_x + dx * 2
                ahead_y = pac_y + dy * 2
                tx = ahead_x + (ahead_x - bx)
                ty = ahead_y + (ahead_y - by)
                tx = max(1, min(W - 2, tx))
                ty = max(1, min(H - 2, ty))
                return tx, ty

            elif self.ghost_id == 3:  # Clyde: chase if far, scatter if close
                gs = ghost_states[self.ghost_id]
                cx, cy = int(gs.getPosition()[0]), int(gs.getPosition()[1])
                dist = abs(pac_x - cx) + abs(pac_y - cy)
                if dist > 8:
                    return pac_x, pac_y
                else:
                    return SCATTER_TARGETS.get(3, (1, 29))

        # Frightened: target doesn't matter (random movement)
        return pac_x, pac_y

    def _legal_dirs(self, gameState):
        """Get list of legal ghost directions from Berkeley's built-in checker."""
        from pacman import GhostRules
        return GhostRules.getLegalActions(gameState, self.index)

    def _choose_direction(self, gameState, tx, ty):
        """Pick direction from legal moves that best approaches target."""
        legal = list(self._legal_dirs(gameState))
        if not legal:
            return Directions.NORTH

        gs = gameState.getGhostStates()[self.ghost_id]
        x, y = int(gs.getPosition()[0]), int(gs.getPosition()[1])

        best_dir = legal[0]
        best_dist = float("inf")
        for d in legal:
            dx, dy = DIR_DELTA[d]
            nx, ny = x + dx, y + dy
            dist = (nx - tx) ** 2 + (ny - ty) ** 2
            if dist < best_dist:
                best_dist = dist
                best_dir = d
        return best_dir

    def _is_legal(self, walls, x, y, d):
        dx, dy = DIR_DELTA[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < walls.width and 0 <= ny < walls.height:
            return not walls[nx][ny]
        return False

    def getAction(self, gameState):
        self._update_pac_dir(gameState)
        mode = self._update_mode(gameState)

        if mode == 2:  # Frightened: random from legal
            legal = list(self._legal_dirs(gameState))
            if legal:
                return np.random.choice(legal)

        # Normal mode: best direction toward target
        tx, ty = self._target(gameState, mode)
        return self._choose_direction(gameState, tx, ty)


class Blinky(OriginalGhost):
    def __init__(self, index): super().__init__(index, 0)

class Pinky(OriginalGhost):
    def __init__(self, index): super().__init__(index, 1)

class Inky(OriginalGhost):
    def __init__(self, index): super().__init__(index, 2)

class Clyde(OriginalGhost):
    def __init__(self, index): super().__init__(index, 3)
