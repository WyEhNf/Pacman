"""Phase B v2: Optimized Double DQN fine-tuning from DAgger_R1.

Optimizations:
- Ring-buffer replay with O(1) sampling (no deque copy)
- Incremental grid updates (only rebuild changed channels)
- 3-stage auto-tuning training loop

Usage:
    python scripts/phaseB_dqn_v2.py
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time, json
from collections import deque

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

# ── Model ──
class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

# ── Layout ──
lo = layout.getLayout('mediumClassic')
TOTAL_FOOD = lo.totalFood

def make_walls():  # precomputed wall channel
    w = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: w[y, x] = 1.0
    return w
WALLS = make_walls()

def state_to_grid(state):
    """Full grid build (used for first frame and eval)."""
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
    """Incremental update: only rebuild dynamic channels (pacman, ghosts, food)."""
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
        self.prev_food_dist = self._food_dist(state, px, py)
        self.prev_capsule_dist = self._capsule_dist(state, px, py)
        self.prev_scared_dist = self._scared_dist(state, px, py)
        self.prev_score = state.getScore()
        self.prev_food_count = state.getFood().count()

    def _food_dist(self, state, px, py):
        fd = state.getFood(); best = 999
        for x in range(fd.width):
            for y in range(fd.height):
                if fd[x][y]:
                    d = abs(px - x) + abs(py - y)
                    if d < best: best = d
        return best if best < 999 else 0

    def _capsule_dist(self, state, px, py):
        caps = state.getCapsules()
        if not caps: return 999
        return min(abs(px - cx) + abs(py - cy) for cx, cy in caps)

    def _scared_dist(self, state, px, py):
        best = 999
        for g in state.getGhostStates():
            if g.scaredTimer > 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                d = abs(px - gx) + abs(py - gy)
                if d < best: best = d
        return best

    def _ghost_dist(self, state, px, py):
        best = 999
        for g in state.getGhostStates():
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            d = abs(px - gx) + abs(py - gy)
            if d < best: best = d
        return best

    def _nonscared_near(self, state, px, py, threshold=6):
        for g in state.getGhostStates():
            if g.scaredTimer <= 0:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                if abs(px - gx) + abs(py - gy) <= threshold:
                    return True
        return False

    def compute(self, state, action_dir, prev_dir, danger_weight=1.0):
        px, py = state.getPacmanPosition()
        curr_score = state.getScore()
        R_base = curr_score - self.prev_score; self.prev_score = curr_score

        gd = self._ghost_dist(state, px, py)
        if gd <= 2:       R_danger = -3.0 * danger_weight
        elif gd <= 4:     R_danger = -1.0 * danger_weight
        elif gd <= 6:     R_danger = -0.3 * danger_weight
        else:              R_danger = 0.0

        R_death = -500.0 if state.isLose() else 0.0

        fd = self._food_dist(state, px, py)
        R_food_nav = 0.0
        if self.prev_food_dist < 999 and fd < 999:
            R_food_nav = np.clip(0.3 * (self.prev_food_dist - fd), -3.0, 3.0)
        self.prev_food_dist = fd

        cfc = state.getFood().count()
        eaten = self.prev_food_count - cfc
        R_food_eaten = 2.0 * eaten if eaten > 0 else 0.0
        self.prev_food_count = cfc
        ghost_killed = (R_base >= 150)

        cd = self._capsule_dist(state, px, py)
        R_capsule = 0.0
        if self._nonscared_near(state, px, py, 6) and cd < 999:
            if self.prev_capsule_dist < 999 and cd < 999:
                R_capsule = np.clip(1.0 * (self.prev_capsule_dist - cd), -3.0, 3.0)
        self.prev_capsule_dist = cd

        sd = self._scared_dist(state, px, py)
        R_chase = 0.0
        if sd < 999:
            if self.prev_scared_dist < 999 and sd < 999:
                R_chase = np.clip(1.5 * (self.prev_scared_dist - sd), -3.0, 3.0)
            elif self.prev_scared_dist >= 999:
                R_chase = 1.5
        self.prev_scared_dist = sd

        R_momentum = 0.1 if (prev_dir and action_dir == prev_dir) else 0.0
        R_win = 200.0 if state.isWin() else 0.0
        R_time = -0.05

        total = R_base + R_danger + R_death + R_food_nav + R_food_eaten + R_capsule + R_chase + R_momentum + R_win + R_time
        return total, ghost_killed

# ── Ghosts ──
PROFILES = {'balanced': (0.5, 0.5), 'aggressive': (0.9, 0.2), 'coward': (0.2, 0.9), 'random': None}
PROF_W = [0.5, 0.2, 0.15, 0.15]

def make_ghosts(p):
    if p == 'random': return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    a, f = PROFILES[p]; return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(lo.getNumGhosts())]

# ── Ring Buffer Replay ──
class RingBuffer:
    def __init__(self, capacity=100000):
        self.cap = capacity
        self.s = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.sn = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.pos = 0; self.size = 0
        # separate indices for death/kill
        self.death_idx = deque(maxlen=5000)
        self.kill_idx = deque(maxlen=5000)

    def push(self, s, a, r, sn, done, is_death, is_kill):
        idx = self.pos
        self.s[idx] = s; self.a[idx] = a; self.r[idx] = r
        self.sn[idx] = sn; self.d[idx] = done
        if is_death: self.death_idx.append(idx)
        if is_kill: self.kill_idx.append(idx)
        self.pos = (self.pos + 1) % self.cap
        if self.size < self.cap: self.size += 1

    def sample(self, batch_size):
        n_kill = min(int(batch_size * 0.1), len(self.kill_idx))
        n_death = min(int(batch_size * 0.2), len(self.death_idx))
        n_main = batch_size - n_kill - n_death

        idx = []
        if n_main > 0 and self.size > 0:
            idx.extend(np.random.randint(0, self.size, n_main))
        if n_death > 0:
            idx.extend(random.sample(list(self.death_idx), n_death))
        if n_kill > 0:
            idx.extend(random.sample(list(self.kill_idx), n_kill))
        random.shuffle(idx)

        return (self.s[idx], self.a[idx], self.r[idx], self.sn[idx], self.d[idx])

    def __len__(self): return self.size

# ── Action selection ──
def select_action(state, q_net, eps, device):
    if random.random() < eps:
        legal = state.getLegalActions(0)
        legal = [a for a in legal if a != Directions.STOP] or legal
        action_dir = random.choice(legal)
        return ACT[action_dir], action_dir
    else:
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

# ── Fast eval ──
def eval_model(q_net, device, n_eps=10, ghost_skill=None):
    scores, wins, foods = [], 0, []
    for _ in range(n_eps):
        if ghost_skill is not None:
            ghosts = [ghostAgents.DirectionalGhost(i + 1, ghost_skill, ghost_skill) for i in range(lo.getNumGhosts())]
        else:
            p = random.choice(list(PROFILES.keys()))
            ghosts = make_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        step = 0
        while not (state.isWin() or state.isLose()) and step < 500:
            mv, ad = select_action(state, q_net, 0.0, device)
            state = state.generateSuccessor(0, ad)
            if state.isWin() or state.isLose(): break
            for gi, gs in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
            step += 1
        scores.append(state.getScore())
        if state.isWin(): wins += 1
        eaten = TOTAL_FOOD - state.getFood().count()
        foods.append(eaten / TOTAL_FOOD * 100)
    return np.mean(scores), wins / n_eps, np.mean(foods)

# ── Training ──
LOG_PATH = os.path.join(PROJECT, 'phaseB_log.txt')
BEST_PATH = os.path.join(PROJECT, 'checkpoints', 'phaseB_dqn_best.pt')
FINAL_PATH = os.path.join(PROJECT, 'checkpoints', 'phaseB_dqn_final.pt')

def log(msg):
    print(msg)
    with open(LOG_PATH, 'a', encoding='utf-8') as f: f.write(msg + '\n')

def train_stage(stage_name, steps, lr, batch, gamma, eps_start, eps_end, eps_decay,
                target_update, train_freq, danger_weight, eval_interval, device):
    log(f'\n{"="*60}')
    log(f'Stage: {stage_name}  Steps: {steps}  LR: {lr}  DangerW: {danger_weight}')
    log(f'Epsilon: {eps_start}->{eps_end}  TargetUpdate: {target_update}')
    log(f'{"="*60}')

    # Load model
    q_net = CNNDQN().to(device)
    if os.path.exists(BEST_PATH):
        q_net.load_state_dict(torch.load(BEST_PATH, map_location=device))
        log('Loaded best model from previous stage.')
    else:
        q_net.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints/dagger_cnn_m0_final.pt'), map_location=device))
        log('Loaded DAgger R1 m0 as starting point.')

    target_net = CNNDQN().to(device); target_net.load_state_dict(q_net.state_dict()); target_net.eval()
    opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=1e-5)
    buffer = RingBuffer()
    shaper = RewardShaper()

    gstep = 0; episode = 0
    ep_scores = deque(maxlen=50); ep_deaths = deque(maxlen=50)
    ep_foods = deque(maxlen=50); ep_kills = deque(maxlen=50)
    loss_hist = deque(maxlen=200); q_hist = deque(maxlen=200)
    best_avg = -1e9
    t0 = time.time()

    # Initial eval
    avg_s, wr, avg_f = eval_model(q_net, device, 10)
    log(f'[INITIAL] score={avg_s:.0f}  win={wr:.1%}  food%={avg_f:.0f}')

    while gstep < steps:
        eps = eps_start + (eps_end - eps_start) * min(1.0, gstep / eps_decay)
        p = random.choices(list(PROFILES.keys()), weights=PROF_W, k=1)[0]
        ghosts = make_ghosts(p)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)

        prev_dir = None; ep_r = 0; ep_s = 0; ep_k = 0
        grid = state_to_grid(state)

        while not (state.isWin() or state.isLose()) and ep_s < 500:
            # Epsilon-greedy with online net
            if random.random() < eps:
                legal = state.getLegalActions(0)
                legal = [a for a in legal if a != Directions.STOP] or legal
                ad = random.choice(legal); mv = ACT[ad]
            else:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(grid).unsqueeze(0).to(device))[0].cpu().numpy()
                legal = state.getLegalActions(0)
                ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
                if not ids: ids = [4]
                best, mv = -1e9, 4
                for i in range(5):
                    if i in ids and q[i] > best: best = q[i]; mv = i
                ad = REV[mv]

            # Pacman step
            state = state.generateSuccessor(0, ad)
            R, killed = shaper.compute(state, ad, prev_dir, danger_weight)
            ep_r += R
            if killed: ep_k += 1

            # Ghosts step
            pacman_died_before = state.isLose()
            if not (state.isWin() or state.isLose()):
                for gi, gs in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)

            done = state.isWin() or state.isLose()
            is_death = state.isLose()
            # If ghost killed pacman after pacman's own move, append death penalty
            if is_death and not pacman_died_before:
                R -= 500.0

            sn_grid = np.zeros((C, H, W), dtype=np.float32) if done else fast_update_grid(grid, state)
            buffer.push(grid, mv, R, sn_grid, done, is_death, killed)
            grid = sn_grid
            prev_dir = ad; ep_s += 1; gstep += 1

            # Train
            if gstep % train_freq == 0 and len(buffer) >= batch:
                s_b, a_b, r_b, sn_b, d_b = buffer.sample(batch)
                s_t = torch.FloatTensor(s_b).to(device)
                a_t = torch.LongTensor(a_b).unsqueeze(1).to(device)
                r_t = torch.FloatTensor(r_b).unsqueeze(1).to(device)
                sn_t = torch.FloatTensor(sn_b).to(device)
                d_t = torch.FloatTensor(d_b).unsqueeze(1).to(device)

                with torch.no_grad():
                    best_a = q_net(sn_t).argmax(dim=1, keepdim=True)
                    q_tgt = target_net(sn_t).gather(1, best_a)
                    target = r_t + gamma * q_tgt * (1 - d_t)

                q_cur = q_net(s_t).gather(1, a_t)
                loss = F.smooth_l1_loss(q_cur, target)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()
                loss_hist.append(loss.item()); q_hist.append(q_cur.mean().item())

            if gstep % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if gstep >= steps: break

        # End of episode
        ep_scores.append(state.getScore()); ep_deaths.append(1 if state.isLose() else 0)
        eaten = TOTAL_FOOD - state.getFood().count()
        ep_foods.append(eaten / TOTAL_FOOD * 100); ep_kills.append(ep_k)
        episode += 1

        # Log
        if episode % 10 == 0:
            l = np.mean(loss_hist) if loss_hist else 0
            q = np.mean(q_hist) if q_hist else 0
            elapsed = time.time() - t0
            eta = (elapsed / gstep * (steps - gstep)) if gstep > 0 else 0
            log(f'  [{gstep:6d}/{steps}] eps={eps:.3f} '
                f'score10={np.mean(ep_scores):7.0f} death10={np.mean(ep_deaths):.2f} '
                f'food10={np.mean(ep_foods):.0f}% kill10={np.mean(ep_kills):.1f} '
                f'loss={l:.4f} Q={q:.2f} |buf|={len(buffer):6d} eta={eta:.0f}s')

        # Eval
        if gstep > 0 and gstep % eval_interval == 0:
            avg_s, wr, avg_f = eval_model(q_net, device, 10)
            log(f'  >>> EVAL@{gstep}: score={avg_s:.0f}  win={wr:.1%}  food%={avg_f:.0f}')
            if avg_s > best_avg:
                best_avg = avg_s
                torch.save(q_net.state_dict(), BEST_PATH)
                log(f'  >>> Best saved: {avg_s:.0f}')

    # End of stage
    final_avg, final_wr, final_f = eval_model(q_net, device, 20)
    log(f'[STAGE END] score={final_avg:.0f}  win={final_wr:.1%}  food%={final_f:.0f}  best={best_avg:.0f}')
    torch.save(q_net.state_dict(), FINAL_PATH)
    return {'score': final_avg, 'win': final_wr, 'food': final_f, 'best': best_avg,
            'death_rate': np.mean(ep_deaths) if ep_deaths else 0}

# ── Main pipeline ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    log(f'Phase B v2 started at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Clear log
    with open(LOG_PATH, 'w', encoding='utf-8') as f: f.write('')

    # ── Stage 1: Warmup ──
    r1 = train_stage(
        stage_name='S1_Warmup', steps=50000, lr=3e-5, batch=256, gamma=0.99,
        eps_start=0.3, eps_end=0.05, eps_decay=5000,
        target_update=2000, train_freq=4, danger_weight=1.0,
        eval_interval=5000, device=device)

    # ── Stage 2: Auto-tune ──
    danger_weight = 1.0
    lr = 3e-5
    batch = 256
    target_update = 2000

    for tune_round in range(3):
        r = train_stage(
            stage_name=f'S2_Tune{tune_round+1}', steps=30000, lr=lr, batch=batch, gamma=0.99,
            eps_start=0.15, eps_end=0.05, eps_decay=3000,
            target_update=target_update, train_freq=4, danger_weight=danger_weight,
            eval_interval=5000, device=device)

        # Auto-tuning logic
        if r['death_rate'] > 0.5:
            danger_weight *= 1.5
            log(f'  [TUNE] Death rate high ({r["death_rate"]:.2f}), danger_weight → {danger_weight:.1f}')
        elif r['score'] < r1['score'] * 0.9:
            lr = max(lr * 0.5, 5e-6)
            log(f'  [TUNE] Score low ({r["score"]:.0f} < {r1["score"]*0.9:.0f}), lr → {lr}')
            target_update = max(target_update // 2, 500)
            log(f'  [TUNE] target_update → {target_update}')
        else:
            log(f'  [TUNE] Metrics acceptable, maintaining params.')
            break

    # ── Stage 3: Final refinement (lower eps, exploit) ──
    r3 = train_stage(
        stage_name='S3_Exploit', steps=20000, lr=lr, batch=batch, gamma=0.99,
        eps_start=0.08, eps_end=0.02, eps_decay=2000,
        target_update=target_update, train_freq=4, danger_weight=danger_weight,
        eval_interval=5000, device=device)

    # ── Final eval ──
    log(f'\n{"="*60}')
    log(f'FINAL BENCHMARK')
    log(f'{"="*60}')
    q_net = CNNDQN().to(device)
    q_net.load_state_dict(torch.load(BEST_PATH, map_location=device))
    for ghost_skill in [0.5, 0.8]:
        s, wr, f = eval_model(q_net, device, 100, ghost_skill=ghost_skill)
        log(f'  Ghost {ghost_skill}: score={s:.0f}  win={wr:.1%}  food%={f:.0f}')

    log(f'\nPhase B complete at {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
