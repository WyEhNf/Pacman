"""Watch best model play smallClassic with 1 random ghost."""
import sys, os, numpy as np, torch, torch.nn as nn

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Game, Directions, Agent
from pacman import ClassicGameRules
from graphicsDisplay import PacmanGraphics
import layout, ghostAgents

H, W, C = 11, 20, 8  # smallClassic is still 20×7, our models expect 20×11

ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
           Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT_MAP.items()}

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

class ModelAgent(Agent):
    def __init__(self, models, layout_h, layout_w):
        super().__init__(0)
        self.models = models
        self.lh, self.lw = layout_h, layout_w

    def registerInitialState(self, state): pass

    def getAction(self, state):
        g = self._state_to_grid(state)
        with torch.no_grad():
            t = torch.FloatTensor(g).unsqueeze(0)
            q = sum(m(t)[0].numpy() for m in self.models) / len(self.models)
        legal = state.getLegalActions(0)
        ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return REV[mv]

    def _state_to_grid(self, state):
        walls = state.getWalls()
        ww, wh = walls.width, walls.height
        g = np.zeros((C, self.lh, self.lw), dtype=np.float32)
        fd = state.getFood()
        for x in range(min(ww, self.lw)):
            for y in range(min(wh, self.lh)):
                if fd[x][y]: g[0, y, x] = 1.0
        for cx, cy in state.getCapsules():
            if 0 <= cx < self.lw and 0 <= cy < self.lh:
                g[1, cy, cx] = 1.0
        px, py = state.getPacmanPosition()
        if 0 <= px < self.lw and 0 <= py < self.lh:
            g[2, py, px] = 1.0
        for i, gh in enumerate(state.getGhostStates()):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < self.lw and 0 <= gy < self.lh:
                g[3 + i, gy, gx] = 1.0
                g[5 + i, gy, gx] = gh.scaredTimer / 40.0
        for x in range(min(ww, self.lw)):
            for y in range(min(wh, self.lh)):
                if walls[x][y]: g[7, y, x] = 1.0
        return g

# Load models
print('Loading models...')
models = []
for ckpt_dir, ids in [('dagger_cnn', ['m0', 'm1', 'm2']),
                        ('cs', ['m0', 'm1', 'm2'])]:
    try:
        for i in range(3):
            m = CNNDQN()
            m.load_state_dict(torch.load(
                os.path.join(PROJECT, f'checkpoints/{ckpt_dir}_{ids[i]}_final.pt'), map_location='cpu'))
            m.eval(); models.append(m)
        print(f'  {ckpt_dir}: {len(models)} models')
        break
    except Exception as e:
        models = []
        continue

if not models:
    for _ in range(3):
        models.append(CNNDQN().eval())
    print('  No checkpoint, using random')

# Play
lo = layout.getLayout('smallClassic')
ghosts = [ghostAgents.RandomGhost(1)]  # 1 random ghost

wall_h, wall_w = lo.walls.height, lo.walls.width
agent = ModelAgent(models, wall_h, wall_w)

print(f'Layout: smallClassic ({wall_w}×{wall_h}), Ghost: random ×1')
print(f'Controls: Space=pause  R=restart  Q=quit\n')

display = PacmanGraphics(zoom=1.5, frameTime=0.1)
rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'\nScore: {game.state.getScore()}  Win: {game.state.isWin()}')
