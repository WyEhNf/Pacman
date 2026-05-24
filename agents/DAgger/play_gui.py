"""GUI Pacman — watch the best model play in real-time.

Controls:
  Space  — pause/resume
  R      — restart new game
  ↑↓     — adjust speed
  Q/ESC  — quit
"""
import sys, os, numpy as np, torch, torch.nn as nn

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Game, Directions, Agent
from pacman import ClassicGameRules
from graphicsDisplay import PacmanGraphics
import layout, ghostAgents

H, W, C = 11, 20, 8
ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
           Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT_MAP.items()}

# ── Model ──
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

# ── Ensemble Agent ──
class EnsembleAgent(Agent):
    def __init__(self, models, layout_obj):
        super().__init__(0)
        self.models = models
        # Walls grid
        self.wg = np.zeros((H, W), dtype=np.float32)
        for x in range(W):
            for y in range(H):
                if layout_obj.walls.data[x][y]:
                    self.wg[y, x] = 1.0

    def registerInitialState(self, state):
        pass

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
        g[7] = self.wg
        return g

    def getAction(self, state):
        grid = self._state_to_grid(state)
        with torch.no_grad():
            t = torch.FloatTensor(grid).unsqueeze(0)
            qs = [m(t)[0].numpy() for m in self.models]
            q = sum(qs) / len(qs)
        legal = state.getLegalActions(0)
        ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return REV[mv]

# ── Load models ──
print('Loading models...')
models = []
checkpoint_dirs = [
    ('selfplay_latest', ['m0', 'm1', 'm2']),  # try selfplay first
    ('sp_r7', ['m0', 'm1', 'm2']),             # fallback r7
    ('dagger_cnn', ['m0', 'm1', 'm2']),        # fallback DAgger
]

loaded = False
for ckpt_dir, ids in checkpoint_dirs:
    try:
        for i in range(3):
            m = CNNDQN()
            path = os.path.join(PROJECT, 'checkpoints', f'{ckpt_dir}_{ids[i]}.pt')
            m.load_state_dict(torch.load(path, map_location='cpu'))
            m.eval(); models.append(m)
        print(f'Loaded from {ckpt_dir}_*.pt ({len(models)} models)')
        loaded = True
        break
    except:
        models = []
        continue

if not loaded:
    print('No checkpoint found, using random model')
    for _ in range(3):
        models.append(CNNDQN().eval())

# ── Ghost config ──
lo = layout.getLayout('mediumClassic')
ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.5, 0.5) for i in range(lo.getNumGhosts())]
agent = EnsembleAgent(models, lo)

# ── Play ──
print(f'\nControls: Space=pause  R=restart  Arrows=speed  Q=quit')
print(f'Ghosts: balanced (0.5/0.5)')
print(f'Layout: mediumClassic\n')

display = PacmanGraphics(zoom=1.2, frameTime=0.08)
rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'\nFinal: score={game.state.getScore()}  win={game.state.isWin()}')
