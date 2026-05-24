"""Evaluate CNN DQN ensemble on mediumClassic."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

H, W, C = 11, 20, 8
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=[2, 3]))

# Precompute walls grid
lo = layout.getLayout('mediumClassic')
W_w, H_w = lo.walls.width, lo.walls.height
walls_grid = np.zeros((H_w, W_w), dtype=np.float32)
for x in range(W_w):
    for y in range(H_w):
        if lo.walls.data[x][y]:
            walls_grid[y, x] = 1.0

def state_to_grid(state):
    """Extract 8-channel grid from GameState."""
    grid = np.zeros((C, H, W), dtype=np.float32)

    # Food
    food = state.getFood()
    for x in range(W):
        for y in range(H):
            if food[x][y]:
                grid[0, y, x] = 1.0

    # Capsules
    for cx, cy in state.getCapsules():
        if 0 <= cx < W and 0 <= cy < H:
            grid[1, cy, cx] = 1.0

    # Pacman
    px, py = state.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H:
        grid[2, py, px] = 1.0

    # Ghosts
    for i, g in enumerate(state.getGhostStates()):
        gx, gy = g.getPosition()
        gx, gy = int(gx), int(gy)
        if 0 <= gx < W and 0 <= gy < H:
            grid[3 + i, gy, gx] = 1.0
            grid[5 + i, gy, gx] = g.scaredTimer / 40.0

    # Walls
    grid[7] = walls_grid
    return grid

# Load models
models = []
for i in range(5):
    m = CNNDQN()
    m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/cnn_m{i}_final.pt'), map_location='cpu'))
    m.eval()
    models.append(m)

ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]

def run_episode(model_fn):
    """model_fn(grid_tensor) → q_values (5,)."""
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        grid = state_to_grid(state)
        q = model_fn(torch.FloatTensor(grid).unsqueeze(0))[0].detach().numpy()
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        masked = {i: q[i] if i in ids else -float('inf') for i in range(5)}
        aid = max(masked, key=masked.get)
        state = state.generateSuccessor(0, REV[aid])
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        step += 1
    return state.getScore(), state.isWin()

# Evaluate individual models
print('─' * 40)
print('Individual CNN Models')
print('─' * 40)
all_scores = []
for i in range(5):
    scores, wins = [], 0
    for ep in range(10):
        s, w = run_episode(models[i])
        scores.append(s); wins += int(w)
    all_scores.extend(scores)
    print(f'CNN {i}: Avg={np.mean(scores):6.0f}  Wins={wins}/10  Min={min(scores):.0f}  Max={max(scores):.0f}')

print(f'Individual Avg: {np.mean(all_scores):.0f}')

# Evaluate Ensemble (average Q-values)
print('\n' + '─' * 40)
print('Ensemble (avg Q-values)')
print('─' * 40)
scores, wins = [], 0
for ep in range(20):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        grid = state_to_grid(state)
        t = torch.FloatTensor(grid).unsqueeze(0)
        # Average Q-values from all models
        q = sum(m(t)[0].detach().numpy() for m in models) / len(models)
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        masked = {i: q[i] if i in ids else -float('inf') for i in range(5)}
        aid = max(masked, key=masked.get)
        state = state.generateSuccessor(0, REV[aid])
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        step += 1
    scores.append(state.getScore())
    if state.isWin(): wins += 1

print(f'Ensemble: Avg={np.mean(scores):.0f}  Wins={wins}/20  Min={min(scores):.0f}  Max={max(scores):.0f}')
