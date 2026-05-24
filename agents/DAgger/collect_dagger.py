"""DAgger Collection: Hybrid model plays, AlphaBeta d3 expert labels every state.

Model explores its own distribution → Expert provides correct actions → Data added to training set.
"""
import sys, os, time, signal, numpy as np, torch, torch.nn as nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState
from multiAgents import AlphaBetaAgent

H, W, C, SEQ = 11, 20, 8, 3
ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
           Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT_MAP.items()}

# ── Model definitions ──
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

# ── Load models ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading Hybrid Ensemble on {device}...')

cnns = []
for i in range(5):
    m = CNNDQN().to(device)
    m.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints', f'dagger_cnn_m{i}_final.pt'), map_location=device))
    m.eval(); cnns.append(m)

# DAgger round 2: only use DAgger CNN models (no GRU)
print(f'{len(cnns)} DAgger CNN models loaded')

# ── Expert ──
expert = AlphaBetaAgent(depth='3', evalFn='betterEvaluationFunction')
print('Expert: AlphaBeta d3 + betterEvaluationFunction')

# ── State → Grid ──
def get_walls_grid(layout_obj):
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if layout_obj.walls.data[x][y]: wg[y, x] = 1.0
    return wg

def state_to_grid(state, wg):
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
    g[7] = wg; return g

def model_q(grid, hist, cnns):
    """DAgger CNN ensemble Q-values (single frame)."""
    with torch.no_grad():
        t = torch.FloatTensor(grid).unsqueeze(0).to(device)
        q = sum(m(t)[0].cpu().numpy() for m in cnns) / len(cnns)
    return q

def pick_action(q, legal):
    ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
    if not ids: ids = [4]
    best, mv = -1e9, 4
    for i in range(5):
        if i in ids and q[i] > best: best = q[i]; mv = i
    return REV[mv]

# ── Ghost factory ──
def make_ghosts(lo, profile='balanced'):
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    attack = {'aggressive': 0.9, 'balanced': 0.5, 'coward': 0.2}[profile]
    flee = {'aggressive': 0.2, 'balanced': 0.5, 'coward': 0.9}[profile]
    return [ghostAgents.DirectionalGhost(i + 1, attack, flee) for i in range(lo.getNumGhosts())]

# ── Main ──
lo = layout.getLayout('mediumClassic')
wg = get_walls_grid(lo)
ghost_profiles = ['aggressive', 'balanced', 'random', 'coward']
num_ghosts = lo.getNumGhosts()

NUM_EPS = 50
all_trajs = []
stopped = False

def on_sig(sig, frame):
    global stopped; stopped = True
    print('\n[Ctrl+C] Stopping...')
signal.signal(signal.SIGINT, on_sig)

print(f'\n{"="*50}')
print(f'  DAgger Collection: {NUM_EPS} episodes')
print(f'  Model: Hybrid (5 CNN + 3 GRU)')
print(f'  Expert: AlphaBeta d3')
print(f'{"="*50}\n')

t0 = time.time()
ep = 0
while ep < NUM_EPS and not stopped:
    ghost_profile = np.random.choice(ghost_profiles)
    ghosts = make_ghosts(lo, ghost_profile)

    st = GameState(); st.initialize(lo, num_ghosts)
    states, actions = [], []
    hist = []
    step = 0
    expert_disagree = 0
    ep_t0 = time.time()

    while not (st.isWin() or st.isLose()) and step < 500:
        # Expert label
        expert_action = expert.getAction(st)

        # Model action
        grid = state_to_grid(st, wg)
        hist.append(grid)
        if len(hist) > SEQ: hist = hist[-SEQ:]
        while len(hist) < SEQ: hist.insert(0, hist[0])
        q = model_q(grid, hist, cnns)
        model_action = pick_action(q, st.getLegalActions(0))

        # Record state with expert label
        states.append(grid)
        actions.append(ACT_MAP[expert_action])

        if model_action != expert_action:
            expert_disagree += 1

        # Advance with MODEL's action (explore model distribution)
        st = st.generateSuccessor(0, model_action)
        if st.isWin() or st.isLose(): break
        for gi, g in enumerate(ghosts):
            if st.isWin() or st.isLose(): break
            st = st.generateSuccessor(gi + 1, g.getAction(st) or Directions.STOP)
        step += 1

    if len(states) < 5:
        continue  # skip very short episodes

    traj = {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.int32),
        'steps': len(states),
        'score': st.getScore(),
        'win': st.isWin(),
        'disagree_rate': expert_disagree / len(states),
        'ghost_profile': ghost_profile,
        'source': 'dagger',
    }
    all_trajs.append(traj)
    dt = time.time() - ep_t0
    ep += 1

    print(f'Ep{ep:3d}/{NUM_EPS} | score={st.getScore():5.0f} win={st.isWin()} '
          f'steps={len(states)} disagree={expert_disagree/len(states)*100:.1f}% '
          f'ghost={ghost_profile[:3]} [{dt:.0f}s]')

    # Save checkpoint every 10 eps
    if ep % 10 == 0:
        ckpt_path = os.path.join(PROJECT, 'data', f'dagger2_{ep:03d}.npz')
        np.savez_compressed(ckpt_path, trajectories=np.array(all_trajs, dtype=object))
        print(f'  [Saved {ckpt_path}]')

elapsed = time.time() - t0
final_path = os.path.join(PROJECT, 'data', 'dagger2_trajectories.npz')
np.savez_compressed(final_path, trajectories=np.array(all_trajs, dtype=object))

scores = [t['score'] for t in all_trajs]
wins = sum(1 for t in all_trajs if t['win'])
disagree = np.mean([t['disagree_rate'] for t in all_trajs])
print(f'\nDone: {ep} eps in {elapsed:.0f}s')
print(f'Avg score: {np.mean(scores):.0f}  Wins: {wins}/{ep}')
print(f'Avg disagree rate: {disagree*100:.1f}%')
print(f'Saved: {final_path}')
