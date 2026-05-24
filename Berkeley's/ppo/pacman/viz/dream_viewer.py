"""Side-by-side visualization: real game vs. world model dream."""
import numpy as np
import pygame
import torch

from pacman.engine.constants import MAZE_ROWS, MAZE_COLS
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel

# Channel-to-color mapping for the 8-channel grid observation
CHANNEL_COLORS = [
    (33, 33, 222),    # Ch 0: Walls (blue)
    (255, 255, 0),    # Ch 1: Pac-Man (yellow)
    (255, 184, 151),  # Ch 2: Pellets (peach)
    (255, 184, 255),  # Ch 3: Power pellets (pink)
    (255, 0, 0),      # Ch 4: Dangerous ghosts (red)
    (33, 33, 255),    # Ch 5: Edible ghosts (blue)
    (40, 40, 40),     # Ch 6: Ghost house (dark)
    (0, 255, 0),      # Ch 7: Fruit (green)
]


def grid_tensor_to_surface(grid: np.ndarray, tile_size: int) -> pygame.Surface:
    """Convert an 8-channel grid (8, 31, 28) numpy array to a pygame surface.

    For each cell, finds the channel with the highest activation (> 0.3 threshold)
    and colors the cell using the corresponding CHANNEL_COLORS entry, scaled by
    activation value.

    Args:
        grid: Numpy array of shape (8, MAZE_ROWS, MAZE_COLS).
        tile_size: Pixel size of each grid cell.

    Returns:
        A pygame.Surface with the rendered grid.
    """
    h = MAZE_ROWS * tile_size
    w = MAZE_COLS * tile_size
    surface = pygame.Surface((w, h))
    surface.fill((0, 0, 0))

    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            activations = grid[:, r, c]
            ch = int(np.argmax(activations))
            val = float(activations[ch])

            if val <= 0.3:
                continue

            alpha = min(1.0, val)
            base_color = CHANNEL_COLORS[ch]
            color = (
                int(base_color[0] * alpha),
                int(base_color[1] * alpha),
                int(base_color[2] * alpha),
            )
            x = c * tile_size
            y = r * tile_size
            pygame.draw.rect(surface, color, (x, y, tile_size, tile_size))

    return surface


class DreamViewer:
    """Side-by-side viewer showing the real Pac-Man game and the world
    model's reconstructed 'dream' synchronized by the same action sequence.

    Controls:
        Q / ESC -- quit
    """

    BG_COLOR = (0, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    LABEL_REAL_COLOR = (100, 255, 100)
    LABEL_DREAM_COLOR = (180, 130, 255)
    GAP = 20       # pixels between the two panels
    INFO_H = 60    # pixels for the info bar at the bottom

    def __init__(
        self,
        world_model: WorldModel,
        config: dict,
        device: torch.device,
        tile_size: int = 16,
    ):
        self.world_model = world_model
        self.config = config
        self.device = device
        self.tile_size = tile_size

        self.maze_w = MAZE_COLS * tile_size
        self.maze_h = MAZE_ROWS * tile_size
        self.width = 2 * self.maze_w + self.GAP
        self.height = self.maze_h + self.INFO_H

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dreaming Pac-Man: Real vs Imagined")
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)
        self.clock = pygame.time.Clock()

        # Environment (use frame_stack=1 config variant for single-frame obs)
        env_config = dict(config)
        env_config["env"] = dict(config["env"])
        env_config["env"]["frame_stack"] = 1
        self.env = PacmanEnv(env_config, difficulty=2)

    def run(self, policy_fn=None, max_steps: int = 3000):
        """Main visualization loop.

        Args:
            policy_fn: Optional callable(obs_dict, legal_mask) -> int action.
                       If None, random legal actions are chosen.
            max_steps: Maximum steps before auto-quit.
        """
        self.world_model.eval()
        obs, _ = self.env.reset(seed=42)
        rng = np.random.default_rng(42)

        # Initialize RSSM state
        h, z = self.world_model.rssm.initial_state(1)

        step = 0
        score = 0
        episode = 1

        running = True
        while running and step < max_steps:
            # --- Handle events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                        break
            if not running:
                break

            # --- Get real observation (single frame) ---
            real_grid = obs["grid"]          # (8, 31, 28)
            real_scalars = obs["scalars"]    # (5,)

            # --- Encode real observation with world model ---
            with torch.no_grad():
                grid_t = torch.as_tensor(
                    real_grid[None], dtype=torch.float32, device=self.device,
                )  # (1, 8, 31, 28)
                scalars_t = torch.as_tensor(
                    real_scalars[None], dtype=torch.float32, device=self.device,
                )  # (1, 5)

                encoder_out = self.world_model.encoder(grid_t, scalars_t)  # (1, 512)

                # Get posterior z from RSSM (uses real observation)
                z, _ = self.world_model.rssm.posterior(h, encoder_out)

                # Decode (h, z) to get the dream grid
                dream_grid_t, _ = self.world_model.decode(h, z)  # (1, 8, 31, 28)
                dream_grid = dream_grid_t[0].cpu().numpy()

            # --- Choose action ---
            legal_mask = self.env.get_legal_mask()
            if policy_fn is not None:
                action = policy_fn(obs, legal_mask)
            else:
                legal_actions = np.where(legal_mask)[0]
                action = int(rng.choice(legal_actions))

            # --- Advance RSSM state using chosen action ---
            with torch.no_grad():
                action_t = torch.tensor([action], dtype=torch.long, device=self.device)
                h = self.world_model.rssm.dynamics(h, z, action_t)

            # --- Step the real environment ---
            obs, reward, terminated, truncated, info = self.env.step(action)
            score = info.get("score", score)
            step += 1

            # --- Compute divergence metric ---
            divergence = float(np.mean((real_grid - dream_grid) ** 2))

            # --- Draw ---
            self.screen.fill(self.BG_COLOR)

            # Left panel: real game
            real_surface = grid_tensor_to_surface(real_grid, self.tile_size)
            self.screen.blit(real_surface, (0, 0))

            # Right panel: dream
            dream_surface = grid_tensor_to_surface(dream_grid, self.tile_size)
            self.screen.blit(dream_surface, (self.maze_w + self.GAP, 0))

            # --- Info bar ---
            info_y = self.maze_h + 5

            # Left label: REAL GAME
            self._draw_text(
                f"REAL GAME  |  Score: {score}  Step: {step}  Ep: {episode}",
                8, info_y, self.font, self.LABEL_REAL_COLOR,
            )

            # Right label: MODEL'S DREAM
            self._draw_text(
                f"MODEL'S DREAM  |  MSE: {divergence:.4f}",
                self.maze_w + self.GAP + 8, info_y, self.font, self.LABEL_DREAM_COLOR,
            )

            # Controls hint
            self._draw_text(
                "Q/ESC = quit",
                8, info_y + 22, self.font_small, (120, 120, 120),
            )

            pygame.display.flip()

            # --- Episode boundary ---
            if terminated:
                obs, _ = self.env.reset()
                h, z = self.world_model.rssm.initial_state(1)
                score = 0
                episode += 1

            self.clock.tick(10)

        pygame.quit()

    def _draw_text(
        self, text: str, x: int, y: int,
        font: pygame.font.Font, color: tuple,
    ):
        """Render text onto the screen."""
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
