"""Watch Hybrid Ensemble play Pacman with GUI."""
import sys, os, numpy as np, torch, torch.nn as nn

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Game, Directions, Agent
from pacman import ClassicGameRules, GameState
from graphicsDisplay import PacmanGraphics
import layout, ghostAgents

H, W, C, SEQ = 11, 20, 8, 3
ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
           Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT_MAP.items()}

# ── Models ──
class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

class GRUDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.v_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))
    def forward(self, x):
        if x.dim() == 4: x = x.unsqueeze(1)
        B, T = x.shape[:2]
        f = self.conv(x.view(B * T, C, H, W)).mean(dim=[2, 3]).view(B, T, 64)
        _, h = self.gru(f); h = h.squeeze(0)
        v, a = self.v_head(h), self.a_head(h)
        return v + a - a.mean(dim=-1, keepdim=True)

# ── Hybrid Agent ──
class HybridAgent(Agent):
    def __init__(self, cnns, grus, walls_grid):
        super().__init__(0)
        self.cnns = cnns
        self.grus = grus
        self.wg = walls_grid
        self.hist = []

    def registerInitialState(self, state):
        self.hist = []

    def _state_to_grid(self, state):
        g = np.zeros((C, H, W), dtype=np.float32)
        fd = state.getFood()
        for x in range(W):
            for y in range(H):
                if fd[x][y]: g[0, y, x] = 1.0
        for cx, cy in state.getCapsules():
            if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
        px, py = state.getPacmanPosition()
        if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0
        for i, gh in enumerate(state.getGhostStates()):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < W and 0 <= gy < H:
                g[3 + i, gy, gx] = 1.0
                g[5 + i, gy, gx] = gh.scaredTimer / 40.0
        g[7] = self.wg; return g

    def getAction(self, state):
        grid = self._state_to_grid(state)
        self.hist.append(grid)
        if len(self.hist) > SEQ: self.hist = self.hist[-SEQ:]
        while len(self.hist) < SEQ: self.hist.insert(0, self.hist[0])

        with torch.no_grad():
            t_cnn = torch.FloatTensor(grid).unsqueeze(0)
            t_gru = torch.FloatTensor(np.stack(self.hist)).unsqueeze(0)
            q_cnn = sum(m(t_cnn)[0].numpy() for m in self.cnns) / len(self.cnns)
            q_gru = sum(m(t_gru)[0].numpy() for m in self.grus) / len(self.grus)
        q = (q_cnn + q_gru) / 2

        legal = state.getLegalActions(0)
        ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return REV[mv]

# ── Load models ──
print('Loading Hybrid Ensemble...')
cnns, grus = [], []
for i in range(5):
    m = CNNDQN()
    m.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints', f'v5_cnn_m{i}_final.pt'), map_location='cpu'))
    m.eval(); cnns.append(m)
for i in [0, 1, 2]:
    m = GRUDQN()
    m.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints', f'gruduel_m{i}_final.pt'), map_location='cpu'))
    m.eval(); grus.append(m)

# ── Play ──
lo = layout.getLayout('mediumClassic')
wg = np.zeros((H, W), dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]: wg[y, x] = 1.0
agent = HybridAgent(cnns, grus, wg)
ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.5, 0.5) for i in range(lo.getNumGhosts())]
display = PacmanGraphics(zoom=1.0, frameTime=0.1)

rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'\nFinal score: {game.state.getScore()}  Win: {game.state.isWin()}')
