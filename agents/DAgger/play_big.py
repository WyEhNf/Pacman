"""Best model plays originalClassic — 4 ghosts, 3 lives, cumulative score."""
import sys, os, numpy as np, torch, torch.nn as nn

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Game, Directions, Agent
from pacman import ClassicGameRules
from graphicsDisplay import PacmanGraphics
import layout, ghostAgents

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}
DIR_VEC = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
           Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}

class CNNDQN(nn.Module):
    def __init__(self, in_c=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=[2, 3]))

class EnsembleAgent(Agent):
    def __init__(self, models, layout_obj):
        super().__init__(0)
        self.models = models
        self.walls = layout_obj.walls
        self.lw, self.lh = self.walls.width, self.walls.height

    def registerInitialState(self, state):
        self.prev_dir = None

    def getAction(self, state):
        g = self._state_to_grid(state)
        with torch.no_grad():
            t = torch.FloatTensor(g).unsqueeze(0)
            q = sum(m(t)[0].numpy() for m in self.models) / len(self.models)

        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]

        # Momentum bias
        if self.prev_dir and ACT[self.prev_dir] in ids:
            q[ACT[self.prev_dir]] += 0.10

        # Food gradient
        px, py = state.getPacmanPosition()
        for act in legal:
            if act == Directions.STOP: continue
            dx, dy = DIR_VEC[act]; cnt = 0
            for d in range(1, 6):
                nx, ny = px + dx * d, py + dy * d
                if 0 <= nx < self.lw and 0 <= ny < self.lh:
                    if not self.walls.data[nx][ny] and state.getFood()[nx][ny]:
                        cnt += 1
            q[ACT[act]] += 0.08 * cnt

        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        self.prev_dir = REV[mv]
        return self.prev_dir

    def _state_to_grid(self, state):
        """Build grid with 8 channels: 0=food,1=capsule,2=pacman,3-4=ghost_pos(best2),5-6=ghost_scared,7=walls"""
        C = 8
        g = np.zeros((C, self.lh, self.lw), dtype=np.float32)

        fd = state.getFood()
        for x in range(self.lw):
            for y in range(self.lh):
                if 0 <= x < fd.width and 0 <= y < fd.height and fd[x][y]:
                    g[0, y, x] = 1.0

        for cx, cy in state.getCapsules():
            if 0 <= cx < self.lw and 0 <= cy < self.lh:
                g[1, cy, cx] = 1.0

        px, py = state.getPacmanPosition()
        if 0 <= px < self.lw and 0 <= py < self.lh:
            g[2, py, px] = 1.0

        # Ghosts: only expose the 2 closest (active) ghosts to the model
        ghosts = state.getGhostStates()
        pacman_pos = (px, py)
        ghost_dists = []
        for gh in ghosts:
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            dist = abs(px - gx) + abs(py - gy)
            ghost_dists.append((dist, gh))

        ghost_dists.sort(key=lambda x: x[0])
        exposed = ghost_dists[:2]  # closest 2 ghosts

        for i, (dist, gh) in enumerate(exposed):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < self.lw and 0 <= gy < self.lh:
                g[3 + i, gy, gx] = 1.0
                g[5 + i, gy, gx] = gh.scaredTimer / 40.0

        # Walls
        for x in range(self.lw):
            for y in range(self.lh):
                if self.walls.data[x][y]:
                    g[7, y, x] = 1.0
        return g

# ── Load models ──
print('Loading DAgger R1 (5 models)...')
models = []
for i in range(5):
    m = CNNDQN()
    m.load_state_dict(torch.load(
        os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location='cpu'))
    m.eval(); models.append(m)

# ── Setup ──
lo = layout.getLayout('originalClassic')
num_g = lo.getNumGhosts()
print(f'Layout: originalClassic ({lo.walls.width}×{lo.walls.height}), {num_g} ghosts')

# 4 ghosts with different strategies
ghosts = [
    ghostAgents.DirectionalGhost(1, 0.9, 0.2),   # aggressive
    ghostAgents.DirectionalGhost(2, 0.5, 0.5),   # balanced
    ghostAgents.DirectionalGhost(3, 0.2, 0.9),   # coward
    ghostAgents.RandomGhost(4),                    # random
]
print('Ghosts: aggressive + balanced + coward + random')

agent = EnsembleAgent(models, lo)

# ── Play ──
print('3 lives, cumulative score\n')
display = PacmanGraphics(zoom=1.0, frameTime=0.06)
rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'\nFinal score: {game.state.getScore()}  Win: {game.state.isWin()}')
