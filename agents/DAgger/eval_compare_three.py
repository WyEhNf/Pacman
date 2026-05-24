"""Compare DAgger_R1, DAgger_bias, Fear_v2 on mediumClassic with identical ghost RNG."""
import sys, os, argparse, numpy as np, torch, torch.nn as nn, random, time

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}
DIR_VEC = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1), Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}

H, W, C = 11, 20, 8
GHOST_S = 0.8  # overridden by --ghost_skill

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
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

# ── Precompute walls ──
lo = layout.getLayout('mediumClassic')
W_w, H_w = lo.walls.width, lo.walls.height
walls_grid = np.zeros((H_w, W_w), dtype=np.float32)
for x in range(W_w):
    for y in range(H_w):
        if lo.walls.data[x][y]: walls_grid[y, x] = 1.0

def state_to_grid(state):
    g = np.zeros((C, H, W), dtype=np.float32)
    fd = state.getFood()
    for x in range(W):
        for y in range(H):
            if x < fd.width and y < fd.height and fd[x][y]: g[0, y, x] = 1.0
    for cx, cy in state.getCapsules():
        if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
    px, py = state.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0

    ghosts = state.getGhostStates()
    ranked = sorted(ghosts, key=lambda gh: abs(px - int(gh.getPosition()[0])) + abs(py - int(gh.getPosition()[1])))
    for i, gh in enumerate(ranked[:2]):
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < W and 0 <= gy < H:
            g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = gh.scaredTimer / 40.0

    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: g[7, y, x] = 1.0
    return g

def load_models(prefix):
    models = []
    for i in range(5):
        m = CNNDQN()
        m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/{prefix}_m{i}_final.pt'), map_location='cpu'))
        m.eval(); models.append(m)
    return models

# ── Evaluation variants ──
def run_episode_dagger_r1(models, seed):
    """Pure DAgger R1 ensemble — no momentum, no food gradient."""
    random.seed(seed); np.random.seed(seed)
    ghosts = [ghostAgents.DirectionalGhost(i + 1, GHOST_S, GHOST_S) for i in range(lo.getNumGhosts())]
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        g = state_to_grid(state)
        t = torch.FloatTensor(g).unsqueeze(0)
        q = sum(m(t)[0].detach().numpy() for m in models) / len(models)
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        state = state.generateSuccessor(0, REV[mv])
        if state.isWin() or state.isLose(): break
        for gi, gs in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
        step += 1
    return state.getScore(), state.isWin()

def run_episode_dagger_bias(models, seed):
    """DAgger R1 + momentum + food gradient bias."""
    random.seed(seed); np.random.seed(seed)
    ghosts = [ghostAgents.DirectionalGhost(i + 1, GHOST_S, GHOST_S) for i in range(lo.getNumGhosts())]
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    prev_dir = None
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        g = state_to_grid(state)
        t = torch.FloatTensor(g).unsqueeze(0)
        q = sum(m(t)[0].detach().numpy() for m in models) / len(models)
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]

        # Momentum bias
        if prev_dir and ACT[prev_dir] in ids:
            q[ACT[prev_dir]] += 0.10

        # Food gradient bias
        px, py = state.getPacmanPosition()
        for act in legal:
            if act == Directions.STOP: continue
            dx, dy = DIR_VEC[act]; cnt = 0
            for d in range(1, 6):
                nx, ny = px + dx * d, py + dy * d
                if 0 <= nx < W and 0 <= ny < H:
                    if not lo.walls.data[nx][ny] and state.getFood()[nx][ny]:
                        cnt += 1
            q[ACT[act]] += 0.08 * cnt

        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        prev_dir = REV[mv]
        state = state.generateSuccessor(0, prev_dir)
        if state.isWin() or state.isLose(): break
        for gi, gs in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
        step += 1
    return state.getScore(), state.isWin()

def run_episode_fear_v2(models, seed):
    """Fear_v2: same as DAgger_bias but with fear2 models."""
    return run_episode_dagger_bias(models, seed)  # Same logic, different checkpoints

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--ghost_skill', type=float, default=0.8, help='Ghost directional skill (0.5=balanced, 0.8=aggressive)')
    args = parser.parse_args()
    N = args.episodes
    global GHOST_S
    GHOST_S = args.ghost_skill

    print(f'Loading models...')
    dagger_models = load_models('dagger_cnn')
    fear2_models = load_models('fear2')
    print(f'DAgger R1: 5 models loaded')
    print(f'Fear v2:   5 models loaded')
    print(f'\nEvaluating on mediumClassic, 2 ghosts (skill={GHOST_S})')
    print(f'Episodes per variant: {N}\n')

    seeds = list(range(N))

    variants = [
        ('DAgger_R1', dagger_models, run_episode_dagger_r1),
        ('DAgger_bias', dagger_models, run_episode_dagger_bias),
        ('Fear_v2', fear2_models, run_episode_fear_v2),
    ]

    results = {}
    for name, models, fn in variants:
        print(f'{"="*50}')
        print(f'Running {name}...')
        print(f'{"="*50}')
        scores, wins = [], 0
        t0 = time.time()
        for ep in range(N):
            s, w = fn(models, seeds[ep])
            scores.append(s); wins += int(w)
            if (ep + 1) % 20 == 0:
                r = np.mean(scores[-20:])
                print(f'  [{ep+1:3d}/{N}] last20_avg={r:7.1f}  win_rate={wins/(ep+1):.2f}')
        elapsed = time.time() - t0
        results[name] = {'scores': np.array(scores), 'wins': wins, 'elapsed': elapsed}
        print(f'  Done in {elapsed:.0f}s')

    # ── Summary ──
    print(f'\n{"="*60}')
    print(f'RESULTS SUMMARY ({N} episodes each, mediumClassic, 2 ghosts skill={GHOST_S})')
    print(f'{"="*60}')
    print(f'{"Variant":<16} {"Avg Score":>10} {"Std":>8} {"Win Rate":>10} {"Min":>8} {"Max":>8}')
    print(f'{"-"*60}')
    for name, res in results.items():
        s = res['scores']
        print(f'{name:<16} {s.mean():10.1f} {s.std():8.1f} {res["wins"]:4d}/{N} ({res["wins"]/N:.0%}) {s.min():8.0f} {s.max():8.0f}')

    # ── Pairwise comparison ──
    print(f'\n{"="*60}')
    print(f'PAIRWISE COMPARISON')
    print(f'{"="*60}')
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diff = results[b]['scores'] - results[a]['scores']
            wins_a = results[a]['wins']; wins_b = results[b]['wins']
            print(f'{b} vs {a}:  Δscore={diff.mean():+.1f}  Δwin={wins_b - wins_a:+d}  '
                  f'better_in={int(np.sum(diff > 0))}/{N} episodes')

if __name__ == '__main__':
    main()
