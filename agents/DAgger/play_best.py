"""Best model (DAgger R1 5-model + momentum + food gradient) plays Pacman GUI.

python scripts/play_best.py              # normal speed
python scripts/play_best.py --slow       # easier to watch
python scripts/play_best.py --big        # originalClassic, 4 ghosts, 3 lives
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--slow', action='store_true')
parser.add_argument('--big', action='store_true')
args = parser.parse_args()

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

C = 8  # channels

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

class BestAgent(Agent):
    def __init__(self, models, layout_obj):
        super().__init__(0)
        self.models = models
        self.walls = layout_obj.walls
        self.W, self.H = layout_obj.walls.width, layout_obj.walls.height

    def registerInitialState(self, s):
        self.prev_dir = None

    def _grid(self, state):
        g = np.zeros((C, self.H, self.W), dtype=np.float32)
        fd = state.getFood()
        for x in range(self.W):
            for y in range(self.H):
                if x < fd.width and y < fd.height and fd[x][y]:
                    g[0, y, x] = 1.0
        for cx, cy in state.getCapsules():
            if 0 <= cx < self.W and 0 <= cy < self.H:
                g[1, cy, cx] = 1.0
        px, py = state.getPacmanPosition()
        if 0 <= px < self.W and 0 <= py < self.H:
            g[2, py, px] = 1.0

        # Only expose 2 closest ghosts to the model
        ghosts = state.getGhostStates()
        px, py = state.getPacmanPosition()
        ranked = sorted(ghosts, key=lambda gh: abs(px-int(gh.getPosition()[0])) + abs(py-int(gh.getPosition()[1])))
        for i, gh in enumerate(ranked[:2]):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < self.W and 0 <= gy < self.H:
                g[3 + i, gy, gx] = 1.0
                g[5 + i, gy, gx] = gh.scaredTimer / 40.0

        for x in range(self.W):
            for y in range(self.H):
                if self.walls.data[x][y]: g[7, y, x] = 1.0
        return g

    def getAction(self, state):
        g = self._grid(state)
        with torch.no_grad():
            t = torch.FloatTensor(g).unsqueeze(0)
            q = sum(m(t)[0].numpy() for m in self.models) / len(self.models)

        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]

        # Momentum
        if self.prev_dir and ACT[self.prev_dir] in ids:
            q[ACT[self.prev_dir]] += 0.10

        # Food gradient
        px, py = state.getPacmanPosition()
        for act in legal:
            if act == Directions.STOP: continue
            dx, dy = DIR_VEC[act]; cnt = 0
            for d in range(1, 6):
                nx, ny = px + dx * d, py + dy * d
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    if not self.walls.data[nx][ny] and state.getFood()[nx][ny]:
                        cnt += 1
            q[ACT[act]] += 0.08 * cnt

        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        self.prev_dir = REV[mv]
        return self.prev_dir

# ── Load ──
print('Loading DAgger R1 (5 models)...')
models = []
for i in range(5):
    m = CNNDQN()
    m.load_state_dict(torch.load(
        os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location='cpu'))
    m.eval(); models.append(m)

# ── Setup ──
if args.big:
    lo = layout.getLayout('originalClassic')
    ghosts = [
        ghostAgents.DirectionalGhost(1, 0.9, 0.2),
        ghostAgents.DirectionalGhost(2, 0.5, 0.5),
        ghostAgents.DirectionalGhost(3, 0.2, 0.9),
        ghostAgents.RandomGhost(4),
    ]
    info = 'originalClassic | 4 ghosts (agg/bal/cow/rand) | 3 lives'
else:
    lo = layout.getLayout('mediumClassic')
    ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.5, 0.5) for i in range(lo.getNumGhosts())]
    info = 'mediumClassic | 2 ghosts (balanced)'

agent = BestAgent(models, lo)
speed = 0.15 if args.slow else 0.06
print(f'DAgger R1 Ensemble (5/5) + momentum + food gradient')
print(f'{info}')
print(f'Speed: {"slow" if args.slow else "fast"}\n')

display = PacmanGraphics(zoom=1.0 if args.big else 1.2, frameTime=speed)
rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'\nScore: {game.state.getScore()}  Win: {game.state.isWin()}')
