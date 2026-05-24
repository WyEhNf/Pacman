"""Phase B v4: DQN from scratch with DAgger-guided exploration.

Key insight: DAgger weights are classifier logits, not Q-values.
Using them as Q-init causes TD divergence.
Fix: Random init DQN, pre-fill buffer with DAgger demos,
use DAgger to guide exploration (80% DAgger, 20% random when exploring).
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

def fast_update_grid(prev_grid, state):
    g = prev_grid.copy()
    g[0].fill(0)
    fd = state.getFood()
    for x in range(W):
        for y in range(H):
            if x < fd.width and y < fd.height and fd[x][y]: g[0, y, x] = 1.0
    g[1].fill(0)
    for cx, cy in state.getCapsules():
        if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
    g[2].fill(0)
    px, py = state.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0
    g[3:7].fill(0)
    ghosts = state.getGhostStates()
    ranked = sorted(ghosts, key=lambda gh: abs(px - int(gh.getPosition()[0])) + abs(py - int(gh.getPosition()[1])))
    for i, gh in enumerate(ranked[:2]):
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < W and 0 <= gy < H:
            g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = gh.scaredTimer / 40.0
    return g

# ── Reward Shaping ──
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
        caps = s.getCapsules()
        if not caps: return 999
        return min(abs(px - cx) + abs(py - cy) for cx, cy in caps)

    def _sd(self, s, px, py):
        b = 999
        for g in s.getGhostStates():
            if g.scaredTimer > 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                b = min(b, abs(px - gx) + abs(py - gy))
        return b

    def _gd(self, s, px, py):
        b = 999
        for g in s.getGhostStates():
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            b = min(b, abs(px - gx) + abs(py - gy))
        return b

    def _ns(self, s, px, py, th=6):
        for g in s.getGhostStates():
            if g.scaredTimer <= 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                if abs(px - gx) + abs(py - gy) <= th: return True
        return False

    def compute(self, state, ad, prev_dir, dw=1.0):
        px, py = state.getPacmanPosition()
        cs = state.getScore(); R_base = cs - self.ps; self.ps = cs
        gd = self._gd(state, px, py)
        if gd <= 2:       R_d = -3.0 * dw
        elif gd <= 4:     R_d = -1.0 * dw
        elif gd <= 6:     R_d = -0.3 * dw
        else:              R_d = 0.0
        R_death = -500.0 if state.isLose() else 0.0
        fd = self._fd(state, px, py); R_fn = 0.0
        if self.pfd < 999 and fd < 999: R_fn = np.clip(0.3 * (self.pfd - fd), -3.0, 3.0)
        self.pfd = fd
        cfc = state.getFood().count(); e = self.pfc - cfc
        R_fe = 2.0 * e if e > 0 else 0.0; self.pfc = cfc
        gk = (R_base >= 150)
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
        return R_base + R_d + R_death + R_fn + R_fe + R_cap + R_ch + R_mom + R_win + R_t, gk

# ── Ghosts ──
PROFILES = {'balanced': (0.5, 0.5), 'aggressive': (0.9, 0.2), 'coward': (0.2, 0.9), 'random': None}
PW = [0.5, 0.2, 0.15, 0.15]

def mk_ghosts(p):
    if p == 'random': return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    a, f = PROFILES[p]; return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(lo.getNumGhosts())]

# ── Ring Buffer ──
class RingBuffer:
    def __init__(self, cap=100000):
        self.cap = cap; self.s = np.zeros((cap, C, H, W), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int32); self.r = np.zeros(cap, dtype=np.float32)
        self.sn = np.zeros((cap, C, H, W), dtype=np.float32); self.d = np.zeros(cap, dtype=np.float32)
        self.pos = 0; self.size = 0
        self.di = deque(maxlen=5000); self.ki = deque(maxlen=5000)

    def push(self, s, a, r, sn, done, dth, kill):
        idx = self.pos; self.s[idx] = s; self.a[idx] = a; self.r[idx] = r
        self.sn[idx] = sn; self.d[idx] = done
        if dth: self.di.append(idx)
        if kill: self.ki.append(idx)
        self.pos = (self.pos + 1) % self.cap
        if self.size < self.cap: self.size += 1

    def sample(self, bs):
        nk = min(int(bs * 0.1), len(self.ki))
        nd = min(int(bs * 0.2), len(self.di))
        nm = bs - nk - nd
        idx = []
        if nm > 0 and self.size > 0: idx.extend(np.random.randint(0, self.size, nm))
        if nd > 0: idx.extend(random.sample(list(self.di), nd))
        if nk > 0: idx.extend(random.sample(list(self.ki), nk))
        random.shuffle(idx)
        return self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx]

    def __len__(self): return self.size

# ── DAgger-guided action ──
def dagger_qvals(state, dagger_models, device):
    """Ensemble Q from DAgger models (for exploration guidance only)."""
    grid = state_to_grid(state)
    t = torch.FloatTensor(grid).unsqueeze(0).to(device)
    q = sum(m(t)[0].cpu().detach().numpy() for m in dagger_models) / len(dagger_models)
    return q

def pick_action(state, q_net, eps, device, dagger_q=None):
    """Epsilon-greedy. When exploring, 80% DAgger suggestion, 20% random.
    If q_net is None, always use DAgger (for pre-fill)."""
    legal = state.getLegalActions(0)
    legal_dirs = [a for a in legal if a != Directions.STOP]
    if not legal_dirs: legal_dirs = [Directions.STOP]

    # If no DQN, always use DAgger
    if q_net is None:
        q = dagger_q if dagger_q is not None else np.zeros(5)
        ids = [ACT[a] for a in legal_dirs] or [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return mv, REV[mv]

    if random.random() < eps:
        # Exploration: 80% DAgger, 20% random
        if dagger_q is not None and random.random() < 0.8:
            q = dagger_q
            ids = [ACT[a] for a in legal_dirs] or [4]
            best, mv = -1e9, 4
            for i in range(5):
                if i in ids and q[i] > best: best = q[i]; mv = i
            return mv, REV[mv]
        else:
            ad = random.choice(legal_dirs); return ACT[ad], ad
    else:
        # Exploitation: use DQN
        grid = state_to_grid(state)
        with torch.no_grad():
            q = q_net(torch.FloatTensor(grid).unsqueeze(0).to(device))[0].cpu().numpy()
        ids = [ACT[a] for a in legal_dirs] or [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return mv, REV[mv]

# ── Eval ──
def evaluate(q_net, device, n=10, gs=None):
    scores, wins, foods = [], 0, []
    for _ in range(n):
        ghosts = [ghostAgents.DirectionalGhost(i + 1, gs, gs) for i in range(lo.getNumGhosts())] if gs else mk_ghosts(random.choice(list(PROFILES.keys())))
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            _, ad = pick_action(state, q_net, 0.0, device)
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

# ── Logging ──
LOG = os.path.join(PROJECT, 'phaseB_v4_log.txt')
BEST = os.path.join(PROJECT, 'checkpoints', 'phaseB_v4_best.pt')
FINAL = os.path.join(PROJECT, 'checkpoints', 'phaseB_v4_final.pt')

def tlog(msg):
    t = time.strftime('%H:%M:%S')
    line = f'[{t}] {msg}'
    print(line)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n')

# ── Pre-fill buffer ──
def prefill_buffer(dagger_models, device, n=100):
    buffer = RingBuffer(); shaper = RewardShaper()
    tlog(f'Pre-filling buffer with {n} DAgger demo episodes...')
    for ep in range(n):
        p = random.choices(list(PROFILES.keys()), weights=PW, k=1)[0]
        ghosts = mk_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)
        grid = state_to_grid(state); pd = None; step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            _, ad = pick_action(state, None, 0.0, device, dagger_q=dagger_qvals(state, dagger_models, device))
            state = state.generateSuccessor(0, ad)
            R, killed = shaper.compute(state, ad, pd)
            pre_d = state.isLose()
            if not (state.isWin() or state.isLose()):
                for gi, gs_ in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs_.getAction(state) or Directions.STOP)
            done = state.isWin() or state.isLose()
            is_d = state.isLose()
            if is_d and not pre_d: R -= 500.0
            sn = np.zeros((C, H, W), dtype=np.float32) if done else fast_update_grid(grid, state)
            buffer.push(grid, ACT[ad], R, sn, done, is_d, killed)
            grid = sn; pd = ad; step += 1
        if (ep + 1) % 25 == 0: tlog(f'  Pre-fill: {ep+1}/{n}  buf={len(buffer)}')
    tlog(f'Pre-fill done: {len(buffer)} transitions.')
    return buffer

# ── Training stage ──
def train_stage(name, steps, lr, batch, gamma, eps_s, eps_e, eps_decay,
                tgt_upd, tfreq, dw, eint, device, buffer, dagger_models):
    tlog(f'\n{"="*55}')
    tlog(f'{name}  steps={steps}  lr={lr}  eps={eps_s}->{eps_e}  dw={dw}')
    tlog(f'{"="*55}')

    # Random init!
    q_net = CNNDQN().to(device)
    if os.path.exists(BEST):
        q_net.load_state_dict(torch.load(BEST, map_location=device))
        tlog('Loaded previous best.')
    else:
        tlog('Random init DQN (not DAgger weights).')
    tgt = CNNDQN().to(device); tgt.load_state_dict(q_net.state_dict()); tgt.eval()
    opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=1e-5)
    shaper = RewardShaper()

    gs = 0; ep = 0
    scr = deque(maxlen=50); dth = deque(maxlen=50)
    fds = deque(maxlen=50); kls = deque(maxlen=50)
    lh = deque(maxlen=200); qh = deque(maxlen=200)
    best_avg = -1e9; t0 = time.time()

    avg_s, wr, avg_f = evaluate(q_net, device, 10)
    tlog(f'[INIT] score={avg_s:.0f}  win={wr:.1%}  food={avg_f:.0f}%')

    while gs < steps:
        eps = eps_s + (eps_e - eps_s) * min(1.0, gs / eps_decay)
        p = random.choices(list(PROFILES.keys()), weights=PW, k=1)[0]
        ghosts = mk_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)
        grid = state_to_grid(state); pd = None; ep_s = 0; ep_k = 0

        while not (state.isWin() or state.isLose()) and ep_s < 500:
            dq = dagger_qvals(state, dagger_models, device)  # for exploration guidance
            mv, ad = pick_action(state, q_net, eps, device, dagger_q=dq)
            state = state.generateSuccessor(0, ad)
            R, killed = shaper.compute(state, ad, pd, dw)
            if killed: ep_k += 1
            pre_d = state.isLose()
            if not (state.isWin() or state.isLose()):
                for gi, gs_ in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs_.getAction(state) or Directions.STOP)
            done = state.isWin() or state.isLose()
            is_d = state.isLose()
            if is_d and not pre_d: R -= 500.0
            sn = np.zeros((C, H, W), dtype=np.float32) if done else fast_update_grid(grid, state)
            buffer.push(grid, mv, R, sn, done, is_d, killed)
            grid = sn; pd = ad; ep_s += 1; gs += 1

            if gs % tfreq == 0 and len(buffer) >= batch:
                sb, ab, rb, snb, db = buffer.sample(batch)
                st = torch.FloatTensor(sb).to(device)
                at = torch.LongTensor(ab).unsqueeze(1).to(device)
                rt = torch.FloatTensor(rb).unsqueeze(1).to(device)
                snt = torch.FloatTensor(snb).to(device)
                dt = torch.FloatTensor(db).unsqueeze(1).to(device)
                with torch.no_grad():
                    ba = q_net(snt).argmax(dim=1, keepdim=True)
                    qtgt = tgt(snt).gather(1, ba)
                    target = rt + gamma * qtgt * (1 - dt)
                qc = q_net(st).gather(1, at)
                loss = F.smooth_l1_loss(qc, target)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                lh.append(loss.item()); qh.append(qc.mean().item())

            if gs % tgt_upd == 0: tgt.load_state_dict(q_net.state_dict())
            if gs >= steps: break

        scr.append(state.getScore()); dth.append(1 if state.isLose() else 0)
        fds.append((TOTAL_FOOD - state.getFood().count()) / TOTAL_FOOD * 100)
        kls.append(ep_k); ep += 1

        if ep % 10 == 0:
            l = np.mean(lh) if lh else 0; q = np.mean(qh) if qh else 0
            et = time.time() - t0
            eta = (et / gs * (steps - gs)) if gs > 0 else 0
            tlog(f'  [{gs:5d}/{steps}] eps={eps:.3f} s10={np.mean(scr):7.0f} d10={np.mean(dth):.2f} '
                 f'f10={np.mean(fds):.0f}% k10={np.mean(kls):.1f} L={l:.3f} Q={q:.2f} '
                 f'buf={len(buffer):5d} eta={eta:.0f}s')

        if gs > 0 and gs % eint == 0:
            avg_s, wr, avg_f = evaluate(q_net, device, 10)
            tlog(f'  >>> EVAL@{gs}: score={avg_s:.0f}  win={wr:.1%}  food={avg_f:.0f}%')
            if avg_s > best_avg:
                best_avg = avg_s; torch.save(q_net.state_dict(), BEST)
                tlog(f'  >>> Best: {avg_s:.0f}')

    fs, fw, ff = evaluate(q_net, device, 20)
    tlog(f'[END] score={fs:.0f}  win={fw:.1%}  food={ff:.0f}%  best={best_avg:.0f}')
    torch.save(q_net.state_dict(), FINAL)
    return {'score': fs, 'win': fw, 'food': ff, 'best': best_avg, 'dr': np.mean(dth) if dth else 0}

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tlog(f'Phase B v4  device={device}  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    open(LOG, 'w').close()

    # Load DAgger ensemble for exploration guidance
    tlog('Loading DAgger R1 ensemble (5 models)...')
    dagger_models = [CNNDQN().to(device) for _ in range(5)]
    for i, m in enumerate(dagger_models):
        m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location=device))
        m.eval()
    tlog('DAgger ensemble loaded.')

    # Pre-fill buffer
    buffer = prefill_buffer(dagger_models, device, 100)

    # ── S1: Learn from DAgger demos ──
    r1 = train_stage('S1_Learn', 40000, lr=1e-4, batch=256, gamma=0.99,
                     eps_s=0.5, eps_e=0.1, eps_decay=10000, tgt_upd=1000,
                     tfreq=4, dw=1.0, eint=5000, device=device,
                     buffer=buffer, dagger_models=dagger_models)

    # ── S2: Fine-tune ──
    r2 = train_stage('S2_Fine', 30000, lr=5e-5, batch=256, gamma=0.99,
                     eps_s=0.15, eps_e=0.05, eps_decay=5000, tgt_upd=1000,
                     tfreq=4, dw=1.0, eint=5000, device=device,
                     buffer=buffer, dagger_models=dagger_models)

    # ── S3: Exploit ──
    train_stage('S3_Exploit', 20000, lr=2e-5, batch=256, gamma=0.99,
                eps_s=0.05, eps_e=0.02, eps_decay=3000, tgt_upd=1000,
                tfreq=4, dw=1.0, eint=5000, device=device,
                buffer=buffer, dagger_models=dagger_models)

    # ── Final eval ──
    tlog(f'\n{"="*55}')
    tlog(f'FINAL BENCHMARK')
    tlog(f'{"="*55}')
    qb = CNNDQN().to(device)
    if os.path.exists(BEST):
        qb.load_state_dict(torch.load(BEST, map_location=device))
    for gs in [0.5, 0.8]:
        s, wr, f = evaluate(qb, device, 100, gs=gs)
        tlog(f'  Ghost {gs}: score={s:.0f}  win={wr:.1%}  food={f:.0f}%')

    tlog(f'Done. {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
