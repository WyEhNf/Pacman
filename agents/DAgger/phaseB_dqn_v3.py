"""Phase B v3: DQN fine-tuning with DAgger buffer pre-fill to prevent policy collapse.

Key fixes from v2:
- Pre-fill replay buffer with 50 DAgger demo episodes (epsilon=0)
- Epsilon: 0.1 -> 0.02 (much lower, policy is already decent)
- LR: 1e-5 (was 3e-5, too aggressive)
- Target update: every 500 steps (was 2000, Q was drifting)
- Training only starts after buffer has >= 2000 transitions
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
        self.prev_food_dist = self._fd(state, px, py)
        self.prev_capsule_dist = self._cd(state, px, py)
        self.prev_scared_dist = self._sd(state, px, py)
        self.prev_score = state.getScore()
        self.prev_food_count = state.getFood().count()

    def _fd(self, s, px, py):
        fd = s.getFood(); best = 999
        for x in range(fd.width):
            for y in range(fd.height):
                if fd[x][y]: d = abs(px - x) + abs(py - y); best = min(best, d)
        return best if best < 999 else 0

    def _cd(self, s, px, py):
        caps = s.getCapsules();
        if not caps: return 999
        return min(abs(px - cx) + abs(py - cy) for cx, cy in caps)

    def _sd(self, s, px, py):
        best = 999
        for g in s.getGhostStates():
            if g.scaredTimer > 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                best = min(best, abs(px - gx) + abs(py - gy))
        return best

    def _gd(self, s, px, py):
        best = 999
        for g in s.getGhostStates():
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            best = min(best, abs(px - gx) + abs(py - gy))
        return best

    def _ns_near(self, s, px, py, th=6):
        for g in s.getGhostStates():
            if g.scaredTimer <= 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                if abs(px - gx) + abs(py - gy) <= th: return True
        return False

    def compute(self, state, ad, prev_dir, danger_w=1.0):
        px, py = state.getPacmanPosition()
        cs = state.getScore(); R_base = cs - self.prev_score; self.prev_score = cs

        gd = self._gd(state, px, py)
        if gd <= 2:       R_danger = -3.0 * danger_w
        elif gd <= 4:     R_danger = -1.0 * danger_w
        elif gd <= 6:     R_danger = -0.3 * danger_w
        else:              R_danger = 0.0

        R_death = -500.0 if state.isLose() else 0.0

        fd = self._fd(state, px, py)
        R_food_nav = 0.0
        if self.prev_food_dist < 999 and fd < 999:
            R_food_nav = np.clip(0.3 * (self.prev_food_dist - fd), -3.0, 3.0)
        self.prev_food_dist = fd

        cfc = state.getFood().count(); eaten = self.prev_food_count - cfc
        R_food_eaten = 2.0 * eaten if eaten > 0 else 0.0
        self.prev_food_count = cfc
        ghost_killed = (R_base >= 150)

        cd = self._cd(state, px, py); R_capsule = 0.0
        if self._ns_near(state, px, py) and cd < 999:
            if self.prev_capsule_dist < 999 and cd < 999:
                R_capsule = np.clip(1.0 * (self.prev_capsule_dist - cd), -3.0, 3.0)
        self.prev_capsule_dist = cd

        sd = self._sd(state, px, py); R_chase = 0.0
        if sd < 999:
            if self.prev_scared_dist < 999 and sd < 999:
                R_chase = np.clip(1.5 * (self.prev_scared_dist - sd), -3.0, 3.0)
            elif self.prev_scared_dist >= 999: R_chase = 1.5
        self.prev_scared_dist = sd

        R_momentum = 0.1 if (prev_dir and ad == prev_dir) else 0.0
        R_win = 200.0 if state.isWin() else 0.0
        R_time = -0.05
        return (R_base + R_danger + R_death + R_food_nav + R_food_eaten + R_capsule + R_chase
                + R_momentum + R_win + R_time), ghost_killed

# ── Ghosts ──
PROFILES = {'balanced': (0.5, 0.5), 'aggressive': (0.9, 0.2), 'coward': (0.2, 0.9), 'random': None}
PROF_W = [0.5, 0.2, 0.15, 0.15]

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
        self.death_idx = deque(maxlen=5000); self.kill_idx = deque(maxlen=5000)

    def push(self, s, a, r, sn, done, is_death, is_kill):
        idx = self.pos; self.s[idx] = s; self.a[idx] = a; self.r[idx] = r
        self.sn[idx] = sn; self.d[idx] = done
        if is_death: self.death_idx.append(idx)
        if is_kill: self.kill_idx.append(idx)
        self.pos = (self.pos + 1) % self.cap
        if self.size < self.cap: self.size += 1

    def sample(self, bs):
        nk = min(int(bs * 0.1), len(self.kill_idx))
        nd = min(int(bs * 0.2), len(self.death_idx))
        nm = bs - nk - nd
        idx = []
        if nm > 0 and self.size > 0: idx.extend(np.random.randint(0, self.size, nm))
        if nd > 0: idx.extend(random.sample(list(self.death_idx), nd))
        if nk > 0: idx.extend(random.sample(list(self.kill_idx), nk))
        random.shuffle(idx)
        return self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx]

    def __len__(self): return self.size

# ── Action selection ──
def pick_action(state, q_net, eps, device):
    if random.random() < eps:
        legal = state.getLegalActions(0)
        legal = [a for a in legal if a != Directions.STOP] or legal
        ad = random.choice(legal); return ACT[ad], ad
    grid = state_to_grid(state)
    with torch.no_grad():
        q = q_net(torch.FloatTensor(grid).unsqueeze(0).to(device))[0].cpu().numpy()
    legal = state.getLegalActions(0)
    ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
    if not ids: ids = [4]
    best, mv = -1e9, 4
    for i in range(5):
        if i in ids and q[i] > best: best = q[i]; mv = i
    return mv, REV[mv]

# ── Eval ──
def evaluate(q_net, device, n=10, ghost_skill=None):
    scores, wins, foods = [], 0, []
    for _ in range(n):
        ghosts = [ghostAgents.DirectionalGhost(i + 1, ghost_skill, ghost_skill) for i in range(lo.getNumGhosts())] if ghost_skill else mk_ghosts(random.choice(list(PROFILES.keys())))
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            _, ad = pick_action(state, q_net, 0.0, device)
            state = state.generateSuccessor(0, ad)
            if state.isWin() or state.isLose(): break
            for gi, gs in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
            step += 1
        scores.append(state.getScore())
        if state.isWin(): wins += 1
        foods.append((TOTAL_FOOD - state.getFood().count()) / TOTAL_FOOD * 100)
    return np.mean(scores), wins / n, np.mean(foods)

# ── Logging ──
LOG = os.path.join(PROJECT, 'phaseB_v3_log.txt')
BEST = os.path.join(PROJECT, 'checkpoints', 'phaseB_v3_best.pt')
FINAL = os.path.join(PROJECT, 'checkpoints', 'phaseB_v3_final.pt')

def tlog(msg):
    print(msg);
    with open(LOG, 'a', encoding='utf-8') as f: f.write(msg + '\n')

# ── Pre-fill buffer with DAgger demos ──
def prefill_buffer(q_net, device, n_eps=50):
    buffer = RingBuffer(); shaper = RewardShaper()
    tlog(f'Pre-filling buffer with {n_eps} DAgger demo episodes...')
    for ep in range(n_eps):
        p = random.choices(list(PROFILES.keys()), weights=PROF_W, k=1)[0]
        ghosts = mk_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)
        grid = state_to_grid(state); prev_dir = None; step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            _, ad = pick_action(state, q_net, 0.0, device)  # epsilon=0, pure exploit
            state = state.generateSuccessor(0, ad)
            R, killed = shaper.compute(state, ad, prev_dir)
            pre_death = state.isLose()
            if not (state.isWin() or state.isLose()):
                for gi, gs in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
            done = state.isWin() or state.isLose()
            is_death = state.isLose()
            if is_death and not pre_death: R -= 500.0
            sn = np.zeros((C, H, W), dtype=np.float32) if done else fast_update_grid(grid, state)
            buffer.push(grid, ACT[ad], R, sn, done, is_death, killed)
            grid = sn; prev_dir = ad; step += 1
        if (ep + 1) % 20 == 0: tlog(f'  Pre-fill: {ep+1}/{n_eps}  buffer size: {len(buffer)}')
    tlog(f'Pre-fill done: {len(buffer)} transitions in buffer.')
    return buffer

# ── Training ──
def train_stage(name, steps, lr, batch, gamma, eps_s, eps_e, eps_decay,
                tgt_upd, train_freq, danger_w, eval_int, device, buffer):
    tlog(f'\n{"="*55}')
    tlog(f'Stage: {name}  Steps: {steps}  LR: {lr}  Eps: {eps_s}->{eps_e}  DangerW: {danger_w}')
    tlog(f'{"="*55}')

    q_net = CNNDQN().to(device)
    if os.path.exists(BEST):
        q_net.load_state_dict(torch.load(BEST, map_location=device)); tlog('Loaded best model.')
    else:
        q_net.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints/dagger_cnn_m0_final.pt'), map_location=device)); tlog('Loaded DAgger m0.')

    tgt_net = CNNDQN().to(device); tgt_net.load_state_dict(q_net.state_dict()); tgt_net.eval()
    opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=1e-5)
    shaper = RewardShaper()

    gs = 0; ep = 0
    ep_scores = deque(maxlen=50); ep_deaths = deque(maxlen=50); ep_foods = deque(maxlen=50); ep_kills = deque(maxlen=50)
    loss_h = deque(maxlen=200); q_h = deque(maxlen=200)
    best_avg = -1e9; t0 = time.time()

    avg_s, wr, avg_f = evaluate(q_net, device, 10)
    tlog(f'[INITIAL] score={avg_s:.0f}  win={wr:.1%}  food%={avg_f:.0f}')

    while gs < steps:
        eps = eps_s + (eps_e - eps_s) * min(1.0, gs / eps_decay)
        p = random.choices(list(PROFILES.keys()), weights=PROF_W, k=1)[0]
        ghosts = mk_ghosts(p); state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)
        grid = state_to_grid(state); prev_dir = None; ep_r = 0; ep_s = 0; ep_k = 0

        while not (state.isWin() or state.isLose()) and ep_s < 500:
            mv, ad = pick_action(state, q_net, eps, device)
            state = state.generateSuccessor(0, ad)
            R, killed = shaper.compute(state, ad, prev_dir, danger_w)
            ep_r += R
            if killed: ep_k += 1
            pre_death = state.isLose()
            if not (state.isWin() or state.isLose()):
                for gi, gs_ in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs_.getAction(state) or Directions.STOP)
            done = state.isWin() or state.isLose()
            is_death = state.isLose()
            if is_death and not pre_death: R -= 500.0
            sn = np.zeros((C, H, W), dtype=np.float32) if done else fast_update_grid(grid, state)
            buffer.push(grid, mv, R, sn, done, is_death, killed)
            grid = sn; prev_dir = ad; ep_s += 1; gs += 1

            if gs % train_freq == 0 and len(buffer) >= batch:
                sb, ab, rb, snb, db = buffer.sample(batch)
                st = torch.FloatTensor(sb).to(device)
                at = torch.LongTensor(ab).unsqueeze(1).to(device)
                rt = torch.FloatTensor(rb).unsqueeze(1).to(device)
                snt = torch.FloatTensor(snb).to(device)
                dt = torch.FloatTensor(db).unsqueeze(1).to(device)
                with torch.no_grad():
                    ba = q_net(snt).argmax(dim=1, keepdim=True)
                    qtgt = tgt_net(snt).gather(1, ba)
                    target = rt + gamma * qtgt * (1 - dt)
                qc = q_net(st).gather(1, at)
                loss = F.smooth_l1_loss(qc, target)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                loss_h.append(loss.item()); q_h.append(qc.mean().item())

            if gs % tgt_upd == 0: tgt_net.load_state_dict(q_net.state_dict())
            if gs >= steps: break

        ep_scores.append(state.getScore()); ep_deaths.append(1 if state.isLose() else 0)
        ep_foods.append((TOTAL_FOOD - state.getFood().count()) / TOTAL_FOOD * 100)
        ep_kills.append(ep_k); ep += 1

        if ep % 10 == 0:
            l = np.mean(loss_h) if loss_h else 0; q = np.mean(q_h) if q_h else 0
            et = time.time() - t0
            eta = (et / gs * (steps - gs)) if gs > 0 else 0
            tlog(f'  [{gs:5d}/{steps}] eps={eps:.3f} s10={np.mean(ep_scores):7.0f} '
                 f'd10={np.mean(ep_deaths):.2f} f10={np.mean(ep_foods):.0f}% '
                 f'k10={np.mean(ep_kills):.1f} L={l:.3f} Q={q:.2f} '
                 f'buf={len(buffer):5d} eta={eta:.0f}s')

        if gs > 0 and gs % eval_int == 0:
            avg_s, wr, avg_f = evaluate(q_net, device, 10)
            tlog(f'  >>> EVAL@{gs}: score={avg_s:.0f}  win={wr:.1%}  food%={avg_f:.0f}')
            if avg_s > best_avg:
                best_avg = avg_s; torch.save(q_net.state_dict(), BEST)
                tlog(f'  >>> Best saved: {avg_s:.0f}')

    final_s, final_wr, final_f = evaluate(q_net, device, 20)
    tlog(f'[END] score={final_s:.0f}  win={final_wr:.1%}  food%={final_f:.0f}  best={best_avg:.0f}')
    torch.save(q_net.state_dict(), FINAL)
    return {'score': final_s, 'win': final_wr, 'food': final_f, 'best': best_avg,
            'death_rate': np.mean(ep_deaths) if ep_deaths else 0}

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tlog(f'Device: {device}  Phase B v3  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    open(LOG, 'w').close()  # clear

    # Load DAgger model for pre-fill
    q0 = CNNDQN().to(device)
    q0.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints/dagger_cnn_m0_final.pt'), map_location=device))
    q0.eval()

    # Pre-fill buffer
    buffer = prefill_buffer(q0, device, 50)

    # ── S1: Conservative warmup ──
    r1 = train_stage('S1_Warmup', 40000, lr=1e-5, batch=256, gamma=0.99,
                     eps_s=0.1, eps_e=0.03, eps_decay=4000, tgt_upd=500,
                     train_freq=4, danger_w=1.0, eval_int=4000, device=device, buffer=buffer)

    # ── S2: Tuning (2 rounds) ──
    lr, dw, tu = 1e-5, 1.0, 500
    for rnd in range(2):
        r = train_stage(f'S2_Tune{rnd+1}', 25000, lr=lr, batch=256, gamma=0.99,
                        eps_s=0.05, eps_e=0.02, eps_decay=2000, tgt_upd=tu,
                        train_freq=4, danger_w=dw, eval_int=4000, device=device, buffer=buffer)
        if r['death_rate'] > 0.6:
            dw *= 1.5; tlog(f'  [TUNE] Death rate {r["death_rate"]:.2f}, danger_w -> {dw:.1f}')
        elif r['score'] < r1['score'] * 0.85:
            lr = max(lr * 0.5, 3e-6); tu = max(tu // 2, 250)
            tlog(f'  [TUNE] Score {r["score"]:.0f} < target, lr->{lr} tu->{tu}')
        else:
            tlog(f'  [TUNE] Ok, stopping tuning.'); break

    # ── S3: Exploit ──
    train_stage('S3_Exploit', 15000, lr=lr, batch=256, gamma=0.99,
                eps_s=0.03, eps_e=0.01, eps_decay=1500, tgt_upd=tu,
                train_freq=4, danger_w=dw, eval_int=4000, device=device, buffer=buffer)

    # ── Final benchmark ──
    tlog(f'\n{"="*55}')
    tlog(f'FINAL BENCHMARK')
    tlog(f'{"="*55}')
    qb = CNNDQN().to(device)
    if os.path.exists(BEST):
        qb.load_state_dict(torch.load(BEST, map_location=device))
    for gs in [0.5, 0.8]:
        s, wr, f = evaluate(qb, device, 100, ghost_skill=gs)
        tlog(f'  Ghost {gs}: score={s:.0f}  win={wr:.1%}  food%={f:.0f}')

    tlog(f'\nDone at {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
