"""Watch CSCNN (Conditional Strategy) model play Pacman with GUI.

python scripts/play_cscnn_gui.py              # default speed
python scripts/play_cscnn_gui.py --slow       # slower, easier to watch
python scripts/play_cscnn_gui.py --model 1    # use a different model
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--slow', action='store_true')
parser.add_argument('--model', type=int, default=0)
args = parser.parse_args()

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Game, Directions, Agent
from pacman import ClassicGameRules
from graphicsDisplay import PacmanGraphics
import layout, ghostAgents

H, W, C = 11, 20, 8
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

lo = layout.getLayout('mediumClassic')
wg = np.zeros((H, W), dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]: wg[y, x] = 1.0

def soft_mode(grid):
    pac = np.argwhere(grid[2] > 0.5)
    if len(pac) == 0: return np.array([0, 1, 0], dtype=np.float32)
    py, px = pac[0]; md, ms = 999.0, 0.0
    for gi in range(2):
        ms = max(ms, grid[5 + gi].max())
        gp = np.argwhere(grid[3 + gi] > 0.5)
        if len(gp) > 0:
            d = abs(py - gp[0][0]) + abs(px - gp[0][1])
            if d < md: md = d
    danger = np.clip(5.0 / max(md, 0.5), 0, 1)
    hunt = np.clip(ms / 40.0, 0, 1)
    if hunt > 0.1: danger *= 1 - hunt
    safe = np.clip(1 - danger - hunt, 0, 1)
    return np.array([safe, danger, hunt], dtype=np.float32)

class CSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc_safe = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
        self.fc_danger = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
        self.fc_hunt = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, x, mw):
        f = self.conv(x).mean(dim=[2, 3])
        qs = self.fc_safe(f); qd = self.fc_danger(f); qh = self.fc_hunt(f)
        w = mw.unsqueeze(-1)
        return (torch.stack([qs, qd, qh], dim=1) * w).sum(dim=1)

DIR_VEC = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
           Directions.EAST: (1, 0), Directions.WEST: (-1, 0), Directions.STOP: (0, 0)}

class CS_Agent(Agent):
    def __init__(self, model):
        super().__init__(0); self.model = model
        self.prev_dir = None

    def registerInitialState(self, s):
        self.prev_dir = None

    def _s2g(self, state):
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
                g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = gh.scaredTimer / 40.0
        g[7] = wg; return g

    def getAction(self, state):
        g = self._s2g(state); mw = soft_mode(g)
        with torch.no_grad():
            q = self.model(torch.FloatTensor(g).unsqueeze(0),
                           torch.FloatTensor(mw).unsqueeze(0))[0].numpy()
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]

        # Direction momentum (prevents oscillation)
        if self.prev_dir and ACT[self.prev_dir] in ids:
            q[ACT[self.prev_dir]] += 0.10

        # Food gradient (navigates toward nearest food)
        px, py = state.getPacmanPosition()
        for act in legal:
            if act == Directions.STOP: continue
            dx, dy = DIR_VEC[act]
            cnt = 0
            for d in range(1, 6):
                nx, ny = px + dx * d, py + dy * d
                if 0 <= nx < W and 0 <= ny < H and not lo.walls.data[nx][ny]:
                    if state.getFood()[nx][ny]: cnt += 1
            q[ACT[act]] += 0.08 * cnt

        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        self.prev_dir = REV[mv]
        return self.prev_dir

# Load model
ckpt = os.path.join(PROJECT, 'checkpoints', f'cs_m{args.model}_final.pt')
m = CSCNN()
m.load_state_dict(torch.load(ckpt, map_location='cpu'))
m.eval()

mode_names = ['SAFE (eat)', 'DANGER (escape)', 'HUNT (chase)']
print(f'CSCNN M{args.model} — 3-mode conditional strategy')
print(f'  {mode_names[0]} / {mode_names[1]} / {mode_names[2]} — auto-detect')
print(f'  Speed: {"slow" if args.slow else "normal"}\n')

agent = CS_Agent(m)
ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.5, 0.5) for i in range(lo.getNumGhosts())]
speed = 0.15 if args.slow else 0.08
display = PacmanGraphics(zoom=1.2, frameTime=speed)
rules = ClassicGameRules()
game = rules.newGame(lo, agent, ghosts, display, quiet=False)
game.run()

print(f'Score: {game.state.getScore()}  Win: {game.state.isWin()}')
