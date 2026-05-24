"""Bridge: Berkeley GameState → PPO model observation → action."""
import sys, os, numpy as np, torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Directions, Agent
from collections import deque

MAZE_ROWS, MAZE_COLS = 31, 28
NUM_CHANNELS = 8
FRAME_STACK = 4

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {0: Directions.NORTH, 1: Directions.SOUTH, 2: Directions.EAST,
       3: Directions.WEST, 4: Directions.STOP}

# Pacman-ai ghost AI direction vectors (4 directions)
DIR_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

class PPOBridgeAgent(Agent):
    def __init__(self, checkpoint_path, config):
        super().__init__(0)
        from pacman.agents.networks import ActorCritic

        env_cfg = config["env"]
        net_cfg = config["network"]
        fs = env_cfg.get("frame_stack", FRAME_STACK)
        grid_c = env_cfg["observation_channels"] * fs

        self.model = ActorCritic(
            grid_channels=grid_c,
            num_scalars=env_cfg.get("num_scalar_features", 5),
            cnn_channels=net_cfg.get("cnn_channels", [32, 64, 64]),
            cnn_kernels=net_cfg.get("cnn_kernels", [3, 3, 3]),
            cnn_strides=net_cfg.get("cnn_strides", [1, 2, 2]),
            shared_hidden=net_cfg.get("shared_hidden", 512),
            head_hidden=net_cfg.get("head_hidden", 128),
        )
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.frame_stack = fs
        self.frame_buffer = deque(maxlen=fs)
        self.prev_dir = None
        self.num_scalars = env_cfg.get("num_scalar_features", 5)
        print(f'PPO model loaded: {checkpoint_path}')

    def registerInitialState(self, state):
        self.frame_buffer.clear()
        obs = self._build_obs(state)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(obs.copy())
        self.prev_dir = None

    def _build_obs(self, state):
        """Build single-frame (NUM_CHANNELS, 31, 28) grid + 5 scalars from Berkeley GameState."""
        grid = np.zeros((NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)
        walls = state.getWalls()

        for r in range(min(MAZE_ROWS, walls.height)):
            for c in range(min(MAZE_COLS, walls.width)):
                if walls[c][r]:
                    grid[0, r, c] = 1.0  # Walls

        food = state.getFood()
        for r in range(min(MAZE_ROWS, food.height)):
            for c in range(min(MAZE_COLS, food.width)):
                if food[c][r]:
                    grid[1, r, c] = 1.0  # Dots

        for cx, cy in state.getCapsules():
            if 0 <= cx < MAZE_COLS and 0 <= cy < MAZE_ROWS:
                grid[2, cy, cx] = 1.0  # Power pellets

        px, py = state.getPacmanPosition()
        if 0 <= px < MAZE_COLS and 0 <= py < MAZE_ROWS:
            grid[3, py, px] = 1.0  # Pac-Man

        # Ghost channels (4+5 for positions, 6 for edible)
        ghosts = state.getGhostStates()
        for i, gh in enumerate(ghosts[:4]):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < MAZE_COLS and 0 <= gy < MAZE_ROWS:
                grid[4 + (i // 2), gy, gx] += 1.0  # Ghost positions (2 ghosts per channel)
                if gh.scaredTimer > 0:
                    grid[6, gy, gx] = 1.0  # Edible ghosts

        # Channel 7: Pac-Man direction one-hot
        if self.prev_dir is not None:
            dr, dc = DIR_DELTA.get(self.prev_dir, (0, 0))
            if 0 <= py + dr < MAZE_ROWS and 0 <= px + dc < MAZE_COLS:
                grid[7, py + dr, px + dc] = 1.0

        # Scalars: lives, level, score, pacman_col, edible_timer
        scalars = np.array([
            3.0,  # lives (hardcoded)
            1.0,  # level
            float(state.getScore()),
            float(px) / MAZE_COLS,
            max(g.scaredTimer for g in ghosts) / 40.0 if ghosts else 0.0,
        ], dtype=np.float32)

        return grid, scalars

    def getAction(self, state):
        grid, scalars = self._build_obs(state)
        self.frame_buffer.append((grid, scalars))

        # Build stacked observation
        grids = [f[0] for f in self.frame_buffer]
        stacked_grid = np.concatenate(grids, axis=0)  # (32, 31, 28)
        stacked_scalars = scalars  # use latest scalars

        with torch.no_grad():
            g = torch.FloatTensor(stacked_grid).unsqueeze(0)
            s = torch.FloatTensor(stacked_scalars).unsqueeze(0)
            logits, _ = self.model(g, s)
            probs = torch.softmax(logits, dim=-1)
            action_idx = probs.argmax(-1).item()

        # Update direction
        if action_idx < 4:
            self.prev_dir = action_idx

        return REV.get(action_idx, Directions.STOP)
