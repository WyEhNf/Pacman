"""Pygame game renderer — draws the maze, entities, and AI sidebar."""

import pygame
import numpy as np

from pacman.engine.constants import (
    Tile, GhostMode, GhostID, GHOST_COLORS,
    MAZE_ROWS, MAZE_COLS, NUM_GHOSTS, Direction,
)
from pacman.engine.entities import GameState
from pacman.viz.sprites import (
    draw_pacman, draw_ghost, draw_pellet, draw_power_pellet, draw_fruit,
)

# Ghost display names
GHOST_NAMES = {
    GhostID.BLINKY: "Blinky",
    GhostID.PINKY: "Pinky",
    GhostID.INKY: "Inky",
    GhostID.CLYDE: "Clyde",
}

ACTION_LABELS = ["UP", "DOWN", "LEFT", "RIGHT"]
MODE_NAMES = {
    GhostMode.SCATTER: "SCT",
    GhostMode.CHASE: "CHS",
    GhostMode.FRIGHTENED: "FRT",
    GhostMode.EATEN: "EAT",
}
MODE_COLORS = {
    GhostMode.SCATTER: (100, 200, 255),
    GhostMode.CHASE: (255, 100, 100),
    GhostMode.FRIGHTENED: (33, 33, 222),
    GhostMode.EATEN: (150, 150, 150),
}


class GameRenderer:
    """Renders the Pac-Man game state using Pygame.

    Controls:
        SPACE — pause/unpause
        N     — advance one step (while paused)
        R     — reset game
        Q     — quit
    """

    BG_COLOR = (0, 0, 0)
    WALL_COLOR = (33, 33, 222)
    GHOST_HOUSE_COLOR = (40, 40, 40)
    GHOST_DOOR_COLOR = (255, 184, 174)
    SIDEBAR_BG = (20, 20, 30)
    TEXT_COLOR = (255, 255, 255)
    LABEL_COLOR = (180, 180, 180)

    def __init__(self, config: dict | None = None,
                 tile_size: int = 20, sidebar_width: int = 240):
        self.tile_size = tile_size
        self.sidebar_width = sidebar_width
        self.maze_width = MAZE_COLS * tile_size
        self.maze_height = MAZE_ROWS * tile_size
        self.width = self.maze_width + sidebar_width
        self.height = self.maze_height
        self.frame = 0
        self.paused = False

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man AI Simulator v2")
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 16)
        self.clock = pygame.time.Clock()

    def render(self, state: GameState,
               agent_info: dict | None = None) -> tuple[bool, bool]:
        """Render one frame.

        Args:
            state: v2 GameState dataclass.
            agent_info: Optional dict with 'action_probs' (ndarray len 4),
                        'value' (float), and optional 'curriculum_phase' (int).

        Returns:
            (running, step_requested): running is False if user quit;
            step_requested is True if user pressed N for a single step.
        """
        step_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False, False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_n:
                    step_requested = True
                if event.key == pygame.K_r:
                    # Signal reset — caller handles actual reset
                    pass

        self.frame += 1
        self.screen.fill(self.BG_COLOR)

        self._draw_maze(state)
        self._draw_entities(state)
        self._draw_sidebar(state, agent_info)

        # Pause indicator
        if self.paused:
            self._draw_text("PAUSED", self.maze_width // 2 - 30,
                            self.maze_height // 2, self.font_large, (255, 255, 0))

        pygame.display.flip()
        return True, step_requested

    def _draw_maze(self, state: GameState):
        """Draw maze walls and pellets from the grid array."""
        ts = self.tile_size
        grid = state.grid
        for r in range(MAZE_ROWS):
            for c in range(MAZE_COLS):
                x = c * ts
                y = r * ts
                tile = grid[r, c]

                if tile == Tile.WALL:
                    pygame.draw.rect(self.screen, self.WALL_COLOR, (x, y, ts, ts))
                    pygame.draw.rect(self.screen, (20, 20, 150), (x, y, ts, ts), 1)
                elif tile == Tile.PELLET:
                    draw_pellet(self.screen, x, y, ts)
                elif tile == Tile.POWER_PELLET:
                    draw_power_pellet(self.screen, x, y, ts, self.frame)
                elif tile == Tile.GHOST_HOUSE:
                    pygame.draw.rect(self.screen, self.GHOST_HOUSE_COLOR, (x, y, ts, ts))
                elif tile == Tile.GHOST_DOOR:
                    pygame.draw.rect(self.screen, self.GHOST_DOOR_COLOR,
                                     (x, y + ts // 2 - 1, ts, 3))

    def _draw_entities(self, state: GameState):
        """Draw Pac-Man, ghosts, and fruit."""
        ts = self.tile_size

        # Fruit
        if state.fruit_active:
            from pacman.engine.maze_data import FRUIT_POSITION
            fr, fc = FRUIT_POSITION
            draw_fruit(self.screen, fc * ts, fr * ts, ts)

        # Ghosts
        for i in range(NUM_GHOSTS):
            if not state.ghost_in_house[i] or state.ghost_exiting[i]:
                gr, gc = state.ghost_pos[i]
                draw_ghost(
                    self.screen,
                    int(gc) * ts, int(gr) * ts, ts,
                    i, GhostMode(state.ghost_mode[i]),
                    int(state.ghost_fright_timer[i]), self.frame,
                )

        # Pac-Man
        pr, pc = state.pac_pos
        draw_pacman(
            self.screen,
            int(pc) * ts, int(pr) * ts, ts,
            int(state.pac_dir), self.frame,
        )

    def _draw_sidebar(self, state: GameState, agent_info: dict | None):
        """Draw info sidebar with game stats and AI info."""
        sx = self.maze_width + 8
        sw = self.sidebar_width - 16

        # Background
        pygame.draw.rect(self.screen, self.SIDEBAR_BG,
                         (self.maze_width, 0, self.sidebar_width, self.height))

        y = 10

        # Title
        self._draw_text("PAC-MAN AI v2", sx, y, self.font_large, (255, 255, 0))
        y += 28

        # --- Game stats ---
        self._draw_text(f"Score:   {state.score}", sx, y, self.font, self.TEXT_COLOR)
        y += 18
        self._draw_text(f"Lives:   {state.pac_lives}", sx, y, self.font, self.TEXT_COLOR)
        y += 18
        self._draw_text(f"Step:    {state.step_count}", sx, y, self.font, self.LABEL_COLOR)
        y += 18
        self._draw_text(f"Pellets: {state.pellets_eaten}/{state.total_pellets}",
                        sx, y, self.font, self.LABEL_COLOR)
        y += 25

        # --- Ghost modes (color-coded) ---
        self._draw_text("GHOST STATUS", sx, y, self.font_large, (100, 200, 255))
        y += 22
        for i in range(NUM_GHOSTS):
            gid = GhostID(i)
            mode = GhostMode(state.ghost_mode[i])
            mode_str = MODE_NAMES.get(mode, "???")
            color = MODE_COLORS.get(mode, self.LABEL_COLOR)
            ghost_color = GHOST_COLORS.get(gid, (255, 255, 255))
            name = GHOST_NAMES.get(gid, f"G{i}")
            # Ghost name in its own color, mode in mode color
            self._draw_text(f"{name:>6}:", sx, y, self.font, ghost_color)
            self._draw_text(mode_str, sx + 80, y, self.font, color)
            y += 16
        y += 15

        # --- Power-up indicator ---
        if state.pac_powered:
            self._draw_text("POWER UP!", sx, y, self.font_large, (0, 255, 255))
            y += 18
            self._draw_text(f"Timer: {state.pac_power_timer}", sx, y,
                            self.font, self.LABEL_COLOR)
            y += 22
        else:
            y += 5

        # --- AI info ---
        if agent_info:
            action_probs = agent_info.get("action_probs")
            value = agent_info.get("value")
            curriculum_phase = agent_info.get("curriculum_phase")

            self._draw_text("AI INFO", sx, y, self.font_large, (100, 200, 255))
            y += 22

            # Action probability bar chart
            if action_probs is not None:
                self._draw_text("Action Probs:", sx, y, self.font, self.LABEL_COLOR)
                y += 16
                bar_max_w = sw - 50
                for idx, label in enumerate(ACTION_LABELS):
                    prob = float(action_probs[idx])
                    bar_w = max(int(prob * bar_max_w), 0)
                    # Label
                    self._draw_text(f"{label:>5}", sx, y, self.font_small, self.LABEL_COLOR)
                    # Bar background
                    bar_x = sx + 42
                    pygame.draw.rect(self.screen, (50, 50, 50),
                                     (bar_x, y + 1, bar_max_w, 12))
                    # Bar fill
                    if bar_w > 0:
                        bar_color = (100, 255, 100) if prob > 0.5 else (200, 200, 100)
                        pygame.draw.rect(self.screen, bar_color,
                                         (bar_x, y + 1, bar_w, 12))
                    # Probability text
                    self._draw_text(f"{prob:.2f}", bar_x + bar_max_w + 4, y,
                                    self.font_small, self.LABEL_COLOR)
                    y += 16
                y += 8

            # V(s) value estimate
            if value is not None:
                self._draw_text(f"V(s): {value:+.2f}", sx, y, self.font, (100, 255, 200))
                y += 20

            # Curriculum phase
            if curriculum_phase is not None:
                self._draw_text(f"Phase: {curriculum_phase}", sx, y,
                                self.font, self.LABEL_COLOR)
                y += 20

        # --- Controls hint ---
        y = self.height - 60
        self._draw_text("Controls:", sx, y, self.font, self.LABEL_COLOR)
        y += 16
        self._draw_text("SPC=pause N=step", sx, y, self.font_small, (120, 120, 120))
        y += 14
        self._draw_text("R=reset  Q=quit", sx, y, self.font_small, (120, 120, 120))

    def _draw_text(self, text: str, x: int, y: int,
                   font: pygame.font.Font, color: tuple):
        """Render text to the screen."""
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def tick(self, fps: int = 10):
        """Limit frame rate."""
        if fps > 0:
            self.clock.tick(fps)

    def close(self):
        """Shut down Pygame."""
        pygame.quit()
