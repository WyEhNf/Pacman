"""Phase B: Reward-Weighted Regression (RWR) fine-tuning of DAgger ensemble.

Simpler than DQN: no Q-function, no TD learning.
1. Run DAgger ensemble to collect episodes with reward shaping
2. Compute return-to-go for each transition
3. Filter positive-return transitions
4. Fine-tune with weighted behavior cloning: weight = clip(return, 0, inf)
5. Iterate

This avoids Q-value divergence entirely while still using the reward signal.
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
from collections import deque

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

H, W, C = 11, 20, 8

class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

lo = layout.getLayout('mediumClassic')
TOTAL_FOOD = lo.totalFood

WALLS = np.zeros((H, W), dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]: WALLS[y, x] = 1.0

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
    g[7] = WALLS; return g

# ── Reward Shaping (from Phase A) ──
class RewardShaper:
    def reset(self, state):
        px, py = state.getPacmanPosition()
        self.pfd = self._fd(state, px, py)
        self.pcd = self._cd(state, px, py)
        self.psd = self._sd(state, px, py)
        self.ps = state.getScore()
        self.pfc = state.getFood().count()

    def _fd(self, s, px, py):
        fd = s.getFood(); b = 999
        for x in range(fd.width):
            for y in range(fd.height):
                if fd[x][y]: b = min(b, abs(px - x) + abs(py - y))
        return b if b < 999 else 0

    def _cd(self, s, px, py):
        caps = s.getCapsules(); return 999 if not caps else min(abs(px - cx) + abs(py - cy) for cx, cy in caps)

    def _sd(self, s, px, py):
        b = 999
        for g in s.getGhostStates():
            if g.scaredTimer > 0:
                b = min(b, abs(px - int(g.getPosition()[0])) + abs(py - int(g.getPosition()[1])))
        return b

    def _gd(self, s, px, py):
        b = 999
        for g in s.getGhostStates():
            b = min(b, abs(px - int(g.getPosition()[0])) + abs(py - int(g.getPosition()[1])))
        return b

    def _ns(self, s, px, py, th=6):
        for g in s.getGhostStates():
            if g.scaredTimer <= 0:
                if abs(px - int(g.getPosition()[0])) + abs(py - int(g.getPosition()[1])) <= th: return True
        return False

    def compute(self, state, ad, prev_dir, dw=1.0):
        px, py = state.getPacmanPosition()
        cs = state.getScore(); R_base = cs - self.ps; self.ps = cs
        gd = self._gd(state, px, py)
        if gd <= 2: R_d = -3.0 * dw
        elif gd <= 4: R_d = -1.0 * dw
        elif gd <= 6: R_d = -0.3 * dw
        else: R_d = 0.0
        R_death = -500.0 if state.isLose() else 0.0
        fd = self._fd(state, px, py); R_fn = 0.0
        if self.pfd < 999 and fd < 999: R_fn = np.clip(0.3 * (self.pfd - fd), -3.0, 3.0)
        self.pfd = fd
        cfc = state.getFood().count(); e = self.pfc - cfc
        R_fe = 2.0 * e if e > 0 else 0.0; self.pfc = cfc
        cd = self._cd(state, px, py); R_cap = 0.0
        if self._ns(state, px, py) and cd < 999:
            if self.pcd < 999 and cd < 999: R_cap = np.clip(1.0 * (self.pcd - cd), -3.0, 3.0)
        self.pcd = cd
        sd = self._sd(state, px, py); R_ch = 0.0
        if sd < 999:
            if self.psd < 999 and sd < 999: R_ch = np.clip(1.5 * (self.psd - sd), -3.0, 3.0)
            elif self.psd >= 999: R_ch = 1.5
        self.psd = sd
        R_mom = 0.1 if (prev_dir and ad == prev_dir) else 0.0
        R_win = 200.0 if state.isWin() else 0.0
        R_t = -0.05
        return R_base + R_d + R_death + R_fn + R_fe + R_cap + R_ch + R_mom + R_win + R_t

# ── Ghosts ──
PROFILES = {'balanced': (0.5, 0.5), 'aggressive': (0.9, 0.2), 'coward': (0.2, 0.9), 'random': None}
PW = [0.5, 0.2, 0.15, 0.15]

def mk_ghosts(p):
    if p == 'random': return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    a, f = PROFILES[p]; return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(lo.getNumGhosts())]

# ── Ensemble action ──
def ensemble_q(models, state, device):
    grid = state_to_grid(state)
    t = torch.FloatTensor(grid).unsqueeze(0).to(device)
    return sum(m(t)[0].cpu().detach().numpy() for m in models) / len(models)

def ensemble_act(models, state, device):
    q = ensemble_q(models, state, device)
    legal = state.getLegalActions(0)
    ids = [ACT[a] for a in legal if a != Directions.STOP] or [4]
    best, mv = -1e9, 4
    for i in range(5):
        if i in ids and q[i] > best: best = q[i]; mv = i
    return mv, REV[mv]

# ── Eval ──
def evaluate(models, device, n=10, gs=None):
    scores, wins, foods = [], 0, []
    for _ in range(n):
        ghosts = [ghostAgents.DirectionalGhost(i + 1, gs, gs) for i in range(lo.getNumGhosts())] if gs else mk_ghosts(random.choice(list(PROFILES.keys())))
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            _, ad = ensemble_act(models, state, device)
            state = state.generateSuccessor(0, ad)
            if state.isWin() or state.isLose(): break
            for gi, gs_ in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi + 1, gs_.getAction(state) or Directions.STOP)
            step += 1
        scores.append(state.getScore())
        if state.isWin(): wins += 1
        foods.append((TOTAL_FOOD - state.getFood().count()) / TOTAL_FOOD * 100)
    return np.mean(scores), wins / n, np.mean(foods)

# ── Collect episodes ──
def collect_episodes(models, device, n_eps, shaper, dw=1.0):
    """Collect episodes using DAgger ensemble (eps=0). Returns list of trajectories."""
    trajs = []
    for ep in range(n_eps):
        p = random.choices(list(PROFILES.keys()), weights=PW, k=1)[0]
        ghosts = mk_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)
        grids, actions, rewards = [], [], []
        pd = None; step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            grids.append(state_to_grid(state))
            _, ad = ensemble_act(models, state, device)
            actions.append(ACT[ad])
            state = state.generateSuccessor(0, ad)
            R = shaper.compute(state, ad, pd, dw)
            rewards.append(R)
            pd = ad
            if state.isWin() or state.isLose(): break
            for gi, gs_ in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi + 1, gs_.getAction(state) or Directions.STOP)
            step += 1
        score = state.getScore()
        # Compute returns-to-go (discounted)
        returns = []
        G = 0
        for r in reversed(rewards): G = r + 0.99 * G; returns.append(G)
        returns.reverse()
        trajs.append({
            'grids': np.array(grids, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int32),
            'returns': np.array(returns, dtype=np.float32),
            'score': score, 'win': state.isWin(), 'steps': len(grids)
        })
    return trajs

# ── Logging ──
LOG = os.path.join(PROJECT, 'phaseB_rwr_log.txt')
BEST_M = os.path.join(PROJECT, 'checkpoints', 'phaseB_rwr_best.pt')

def tlog(msg):
    t = time.strftime('%H:%M:%S')
    line = f'[{t}] {msg}'; print(line)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n')

# ── Training iteration ──
def train_iter(models, trajs, lr, batch, epochs, device):
    """Fine-tune models using reward-weighted behavior cloning.
    Each model trains on a different 80% subset (5-fold split) to preserve ensemble diversity."""
    # Flatten all trajectories, filter positive returns
    all_g, all_a, all_w = [], [], []
    for tr in trajs:
        for i in range(len(tr['grids'])):
            w = max(0.0, tr['returns'][i])
            if w > 0:
                all_g.append(tr['grids'][i]); all_a.append(tr['actions'][i]); all_w.append(w)
    if len(all_g) < batch * 5:
        tlog(f'  Only {len(all_g)} positive-return transitions, skipping.')
        return

    all_g = np.array(all_g, dtype=np.float32); all_a = np.array(all_a, dtype=np.int64); all_w = np.array(all_w, dtype=np.float32)
    all_w = all_w / (all_w.sum() + 1e-8) * len(all_w)
    N = len(all_g)

    results = []
    for mi, model in enumerate(models):
        # Model i trains on all data EXCEPT fold i (preserves diversity)
        fold_size = N // 5
        val_start = mi * fold_size
        val_end = (mi + 1) * fold_size if mi < 4 else N
        train_idx = list(range(0, val_start)) + list(range(val_end, N))
        train_g = all_g[train_idx]; train_a = all_a[train_idx]; train_w = all_w[train_idx]
        Nt = len(train_idx)

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            perm = np.random.permutation(Nt)
            total_loss = 0; n_batches = 0
            for start in range(0, Nt, batch):
                idx = perm[start:start + batch]
                gb = torch.FloatTensor(train_g[idx]).to(device)
                ab = torch.LongTensor(train_a[idx]).to(device)
                wb = torch.FloatTensor(train_w[idx]).to(device)
                logits = model(gb)
                loss = -(F.log_softmax(logits, dim=1).gather(1, ab.unsqueeze(1)).squeeze() * wb).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                total_loss += loss.item(); n_batches += 1
        model.eval()
        results.append(total_loss / max(n_batches, 1))
    return results

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tlog(f'Phase B RWR  device={device}  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    open(LOG, 'w').close()

    # Load DAgger R1 ensemble
    tlog('Loading DAgger R1 ensemble...')
    models = [CNNDQN().to(device) for _ in range(5)]
    for i, m in enumerate(models):
        m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location=device))
        m.eval()
    tlog('5 models loaded.')

    # Baseline eval
    s0, w0, f0 = evaluate(models, device, 50)
    tlog(f'[BASELINE] 50eps: score={s0:.0f}  win={w0:.1%}  food={f0:.0f}%')
    for gs in [0.5, 0.8]:
        s, w, f = evaluate(models, device, 50, gs=gs)
        tlog(f'  Ghost{gs}: score={s:.0f}  win={w:.1%}  food={f:.0f}%')

    # RL iterations
    shaper = RewardShaper()
    best_score = s0
    lr = 1e-5

    for iteration in range(5):
        tlog(f'\n--- Iteration {iteration + 1} ---')

        # Collect episodes
        tlog(f'Collecting 200 episodes...')
        trajs = collect_episodes(models, device, 200, shaper, dw=1.0)
        scores = [t['score'] for t in trajs]; wins = sum(t['win'] for t in trajs)
        pos_ret = sum(1 for t in trajs if max(t['returns']) > 0)
        tlog(f'  Collected: avg_score={np.mean(scores):.0f}  wins={wins}/200  '
             f'pos_ret_eps={pos_ret}  total_trans={sum(len(t["grids"]) for t in trajs)}')

        # Train
        tlog(f'Training (lr={lr}, batch=256, epochs=3)...')
        losses = train_iter(models, trajs, lr=lr, batch=256, epochs=3, device=device)
        tlog(f'  Losses: {[f"{l:.4f}" for l in losses]}')

        # Eval
        s, w, f = evaluate(models, device, 50)
        tlog(f'  Eval50: score={s:.0f}  win={w:.1%}  food={f:.0f}%')
        if s > best_score:
            best_score = s
            for i, m in enumerate(models):
                torch.save(m.state_dict(), os.path.join(PROJECT, 'checkpoints', f'phaseB_rwr_m{i}_best.pt'))
            tlog(f'  Best saved: {s:.0f}')

    # ── Final benchmark ──
    tlog(f'\n{"="*55}')
    tlog(f'FINAL BENCHMARK')
    tlog(f'{"="*55}')
    for gs in [0.5, 0.8]:
        s, w, f = evaluate(models, device, 100, gs=gs)
        tlog(f'  Ghost {gs}: score={s:.0f}  win={w:.1%}  food={f:.0f}%')

    tlog(f'Done. {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
