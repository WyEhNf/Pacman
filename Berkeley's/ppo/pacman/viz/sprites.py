"""Sprite drawing functions for Pac-Man entities using Pygame primitives."""

import math
import pygame
from pacman.engine.constants import (
    GhostMode, GHOST_COLORS, FRIGHTENED_COLOR, FRIGHTENED_FLASH_COLOR,
)


def draw_pacman(surface: pygame.Surface, x: int, y: int, size: int,
                direction: int, frame: int):
    """Draw Pac-Man as a yellow circle with animated mouth."""
    center = (x + size // 2, y + size // 2)
    radius = size // 2 - 1

    # Mouth animation
    mouth_angle = abs(math.sin(frame * 0.3)) * 45  # oscillating mouth

    # Direction to angle mapping
    angle_map = {0: 90, 1: 270, 2: 180, 3: 0}  # UP, DOWN, LEFT, RIGHT
    base_angle = angle_map.get(direction, 0)

    start_angle = math.radians(base_angle + mouth_angle)
    end_angle = math.radians(base_angle + 360 - mouth_angle)

    # Draw filled arc (Pac-Man shape)
    points = [center]
    for angle in range(int(math.degrees(start_angle)), int(math.degrees(end_angle)) + 1, 5):
        rad = math.radians(angle)
        px = center[0] + radius * math.cos(rad)
        py = center[1] - radius * math.sin(rad)
        points.append((px, py))
    points.append(center)

    if len(points) > 2:
        pygame.draw.polygon(surface, (255, 255, 0), points)


def draw_ghost(surface: pygame.Surface, x: int, y: int, size: int,
               ghost_id: int, mode: GhostMode, frightened_timer: int, frame: int):
    """Draw a ghost with appropriate color based on mode."""
    center_x = x + size // 2
    center_y = y + size // 2
    radius = size // 2 - 1

    # Determine color
    if mode == GhostMode.FRIGHTENED:
        # Blink white/blue when timer is low
        if frightened_timer < 12 and frame % 6 < 3:
            color = FRIGHTENED_FLASH_COLOR
        else:
            color = FRIGHTENED_COLOR
    elif mode == GhostMode.EATEN:
        # Draw just eyes
        _draw_ghost_eyes(surface, center_x, center_y, radius)
        return
    else:
        color = GHOST_COLORS.get(ghost_id, (255, 0, 0))

    # Ghost body — rounded top, wavy bottom
    # Top half circle
    pygame.draw.circle(surface, color, (center_x, center_y - 2), radius)
    # Bottom rectangle
    pygame.draw.rect(surface, color,
                     (center_x - radius, center_y - 2, radius * 2, radius))
    # Wavy bottom
    wave_y = center_y + radius - 2
    wave_w = radius * 2 // 3
    for i in range(3):
        wx = center_x - radius + i * wave_w
        if frame % 4 < 2:
            pygame.draw.circle(surface, color, (wx + wave_w // 2, wave_y), wave_w // 2)
        else:
            pygame.draw.circle(surface, color, (wx + wave_w // 2, wave_y + 2), wave_w // 2)

    # Eyes
    _draw_ghost_eyes(surface, center_x, center_y, radius)


def _draw_ghost_eyes(surface: pygame.Surface, cx: int, cy: int, radius: int):
    """Draw ghost eyes."""
    eye_radius = max(radius // 3, 2)
    pupil_radius = max(eye_radius // 2, 1)
    eye_y = cy - radius // 4

    for offset in [-radius // 3, radius // 3]:
        # White of eye
        pygame.draw.circle(surface, (255, 255, 255), (cx + offset, eye_y), eye_radius)
        # Pupil
        pygame.draw.circle(surface, (0, 0, 128), (cx + offset, eye_y), pupil_radius)


def draw_pellet(surface: pygame.Surface, x: int, y: int, size: int):
    """Draw a small pellet dot."""
    center = (x + size // 2, y + size // 2)
    pygame.draw.circle(surface, (255, 255, 200), center, 2)


def draw_power_pellet(surface: pygame.Surface, x: int, y: int, size: int, frame: int):
    """Draw a flashing power pellet."""
    center = (x + size // 2, y + size // 2)
    radius = size // 3
    if frame % 10 < 7:  # flash effect
        pygame.draw.circle(surface, (255, 255, 200), center, radius)


def draw_fruit(surface: pygame.Surface, x: int, y: int, size: int):
    """Draw a fruit (cherry-like)."""
    center = (x + size // 2, y + size // 2)
    pygame.draw.circle(surface, (255, 0, 0), (center[0] - 2, center[1] + 2), size // 4)
    pygame.draw.circle(surface, (255, 0, 0), (center[0] + 2, center[1] + 2), size // 4)
    pygame.draw.line(surface, (0, 180, 0), center, (center[0], center[1] - size // 3), 2)
