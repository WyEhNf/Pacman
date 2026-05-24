"""Phase B v2: Online Double DQN with BC regularization + conservative tuning.

Key fixes from v1 crash:
- Lower epsilon (0.1→0.02) — trust pretrained policy more
- Lower LR (1e-5) — prevent catastrophic forgetting
- BC regularization loss — keep policy close to DAgger teacher
- Warmup phase — fill buffer with DAgger data before training

python scripts/phaseB_online.py
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
from datetime import datetime

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
STOP_HOUR = 7
LOG_PATH = os.path.join(PROJECT, 'data', 'phaseB_log.txt')

# ── Model ──
class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

# ── Layout ──
lo = layout.getLayout('mediumClassic')
WALLS = np.zeros((H, W), dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]: WALLS[y, x] = 1.0
TOTAL_FOOD = lo.totalFood

def state_to_grid(s):
    g = np.zeros((C, H, W), dtype=np.float32)
    fd = s.getFood()
    for x in range(W):
        for y in range(H):
            if x < fd.width and y < fd.height and fd[x][y]: g[0, y, x] = 1.0
    for cx, cy in s.getCapsules():
        if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
    px, py = s.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0
    ghosts = s.getGhostStates()
    ranked = sorted(ghosts, key=lambda gh: abs(px - int(gh.getPosition()[0])) + abs(py - int(gh.getPosition()[1])))
    for i, gh in enumerate(ranked[:2]):
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < W and 0 <= gy < H:
            g[3+i, gy, gx] = 1.0; g[5+i, gy, gx] = gh.scaredTimer / 40.0
    g[7] = WALLS; return g

# ── Reward ──
class RewardShaper:
    def reset(self, s):
        px, py = s.getPacmanPosition()
        self.prev_score = s.getScore()
        self.prev_food = s.getFood().count()
    def compute(self, s, ad, pd):
        px, py = s.getPacmanPosition()
        cs = s.getScore(); R_base = cs - self.prev_score; self.prev_score = cs
        gd = min((abs(px - int(g.getPosition()[0])) + abs(py - int(g.getPosition()[1]))) for g in s.getGhostStates())
        R_danger = -3.0 if gd <= 2 else (-1.0 if gd <= 4 else (-0.3 if gd <= 6 else 0.0))
        R_death = -500.0 if s.isLose() else 0.0
        fe = self.prev_food - s.getFood().count()
        R_food = 2.0 * fe if fe > 0 else 0.0; self.prev_food = s.getFood().count()
        R_mom = 0.1 if (pd and ad == pd) else 0.0
        R_win = 200.0 if s.isWin() else 0.0; R_time = -0.05
        gk = (R_base >= 150)
        return (R_base + R_danger + R_death + R_food + R_mom + R_win + R_time, gk)

# ── Ghosts ──
GT = ['balanced','balanced','balanced','aggressive','coward','random']
def mk_ghosts(p):
    if p == 'random': return [ghostAgents.RandomGhost(i+1) for i in range(lo.getNumGhosts())]
    d = {'aggressive':(0.9,0.2),'balanced':(0.5,0.5),'coward':(0.2,0.9)}[p]
    return [ghostAgents.DirectionalGhost(i+1,d[0],d[1]) for i in range(lo.getNumGhosts())]

# ── Replay Buffer ──
class ReplayBuffer:
    def __init__(self, cap=100000):
        self.b, self.db, self.kb = [], [], []
        self.cap, self.dc, self.kc = cap, 5000, 5000
    def push(self, s, a, r, sn, done, is_death, is_kill):
        e = (s, a, r, sn, done)
        self.b.append(e)
        if len(self.b) > self.cap: self.b = self.b[len(self.b)//4:]
        if is_death: self.db.append(e);
        if len(self.db) > self.dc: self.db = self.db[len(self.db)//4:]
        if is_kill: self.kb.append(e)
        if len(self.kb) > self.kc: self.kb = self.kb[len(self.kb)//4:]
    def sample(self, bs):
        nk = min(int(bs*0.05), len(self.kb))
        nd = min(int(bs*0.10), len(self.db))
        nm = bs - nk - nd
        items = []
        if nm > 0 and self.b: items += [self.b[i] for i in np.random.randint(0, len(self.b), nm)]
        if nd > 0: items += [self.db[i] for i in np.random.randint(0, len(self.db), nd)]
        if nk > 0: items += [self.kb[i] for i in np.random.randint(0, len(self.kb), nk)]
        random.shuffle(items)
        return (np.stack([x[0] for x in items]),
                np.array([x[1] for x in items], dtype=np.int64),
                np.array([x[2] for x in items], dtype=np.float32),
                np.stack([x[3] for x in items]),
                np.array([x[4] for x in items], dtype=np.float32))
    def __len__(self): return len(self.b)

# ── Log ──
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_PATH, 'a') as f: f.write(line+'\n')

# ── DAgger teacher action ──
def dagger_q(grid, models):
    t = torch.FloatTensor(grid).unsqueeze(0)
    return sum(m(t)[0].detach().numpy() for m in models) / len(models)

def dagger_act(grid, models, state):
    q = dagger_q(grid, models)
    legal = state.getLegalActions(0)
    ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
    if not ids: ids = [4]
    return max(ids, key=lambda i: q[i])

# ── Eval ──
def evaluate(qn, dev, n=30):
    sc, wi, fo, de, st = [], 0, [], 0, []
    for _ in range(n):
        p = random.choice(GT); gs = mk_ghosts(p)
        s = GameState(); s.initialize(lo, lo.getNumGhosts()); step = 0
        while not (s.isWin() or s.isLose()) and step < 500:
            g = state_to_grid(s)
            with torch.no_grad():
                q = qn(torch.FloatTensor(g).unsqueeze(0).to(dev))[0].cpu().numpy()
            legal = s.getLegalActions(0)
            ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
            if not ids: ids = [4]
            s = s.generateSuccessor(0, REV[max(ids, key=lambda i: q[i])])
            if s.isWin() or s.isLose(): break
            for gi, gh in enumerate(gs):
                if s.isWin() or s.isLose(): break
                s = s.generateSuccessor(gi+1, gh.getAction(s) or Directions.STOP)
            step += 1
        sc.append(s.getScore())
        if s.isWin(): wi += 1
        if s.isLose(): de += 1
        fo.append((TOTAL_FOOD - s.getFood().count()) / TOTAL_FOOD * 100)
        st.append(step)
    return np.mean(sc), wi/n, np.mean(fo), de/n, np.mean(st)

def eval_baseline(models, bias=False, n=50):
    sc, wi = [], 0
    for _ in range(n):
        p = random.choice(['balanced']*3+['aggressive','coward','random'])
        gs = mk_ghosts(p)
        s = GameState(); s.initialize(lo, lo.getNumGhosts())
        pd = None; step = 0
        while not (s.isWin() or s.isLose()) and step < 500:
            g = state_to_grid(s); q = dagger_q(g, models)
            legal = s.getLegalActions(0)
            ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
            if not ids: ids = [4]
            if bias:
                if pd and ACT[pd] in ids: q[ACT[pd]] += 0.10
                px, py = s.getPacmanPosition()
                for act in legal:
                    if act == Directions.STOP: continue
                    dx, dy = DIR_VEC[act]; cnt = 0
                    for d in range(1, 6):
                        nx, ny = px+dx*d, py+dy*d
                        if 0 <= nx < W and 0 <= ny < H:
                            if not lo.walls.data[nx][ny] and s.getFood()[nx][ny]: cnt += 1
                    q[ACT[act]] += 0.08*cnt
            mv = max(ids, key=lambda i: q[i]); pd = REV[mv]
            s = s.generateSuccessor(0, pd)
            if s.isWin() or s.isLose(): break
            for gi, gh in enumerate(gs):
                if s.isWin() or s.isLose(): break
                s = s.generateSuccessor(gi+1, gh.getAction(s) or Directions.STOP)
            step += 1
        sc.append(s.getScore())
        if s.isWin(): wi += 1
    return np.mean(sc), wi/n

# ── Main ──
def main():
    open(LOG_PATH, 'w').close()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Phase B v2 Online | device={dev} | stop at {STOP_HOUR}:00 | food={TOTAL_FOOD}')

    # Load baselines
    log('Loading models...')
    dagger_m = []; fear2_m = []
    for i in range(5):
        m = CNNDQN(); m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location='cpu')); m.eval(); dagger_m.append(m)
    for i in range(5):
        m = CNNDQN(); m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/fear2_m{i}_final.pt'), map_location='cpu')); m.eval(); fear2_m.append(m)

    log('\n=== BASELINE (50 eps) ===')
    for nm, md, bi in [('DAgger_R1',dagger_m,False),('DAgger_bias',dagger_m,True),('Fear_v2',fear2_m,True)]:
        s, w = eval_baseline(md, bi, 50)
        log(f'  {nm:<16} score={s:7.1f}  win={w:.1%}')

    # Init DQN
    log('\nInit DQN from DAgger...')
    qn = CNNDQN().to(dev)
    qn.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints/dagger_cnn_m0_final.pt'), map_location=dev))
    tn = CNNDQN().to(dev); tn.load_state_dict(qn.state_dict()); tn.eval()
    opt = torch.optim.Adam(qn.parameters(), lr=1e-5, weight_decay=1e-5)
    buf = ReplayBuffer(); rsh = RewardShaper()

    # Initial eval
    s0, w0, f0, d0, st0 = evaluate(qn, dev, 30)
    log(f'[INIT] score={s0:.0f} win={w0:.1%} food%={f0:.0f} death={d0:.1%} steps={st0:.0f}')

    # ── Warmup: fill buffer with DAgger-only data ──
    log('\n=== WARMUP (2k DAgger steps) ===')
    warmup_ep = 0
    while len(buf) < 2000 and warmup_ep < 30:
        p = random.choice(GT); gs = mk_ghosts(p)
        s = GameState(); s.initialize(lo, lo.getNumGhosts()); rsh.reset(s)
        pd = None; step = 0
        while not (s.isWin() or s.isLose()) and step < 200:
            g = state_to_grid(s)
            mv = dagger_act(g, dagger_m, s); ad = REV[mv]
            s = s.generateSuccessor(0, ad)
            R, gk = rsh.compute(s, ad, pd); pd_died = s.isLose()
            if not (s.isWin() or s.isLose()):
                for gi, gh in enumerate(gs):
                    if s.isWin() or s.isLose(): break
                    s = s.generateSuccessor(gi+1, gh.getAction(s) or Directions.STOP)
            done = s.isWin() or s.isLose()
            ng = np.zeros((C,H,W), dtype=np.float32) if done else state_to_grid(s)
            if s.isLose() and not pd_died: R -= 500.0
            buf.push(g, mv, R, ng, done, s.isLose(), gk)
            pd = ad; step += 1
        warmup_ep += 1
    log(f'  Warmup done: {warmup_ep} eps, {len(buf)} transitions')

    # Hyperparams
    BATCH, GAMMA = 256, 0.99
    TGT_UPD, TRAIN_FREQ = 1000, 4
    EPS_S, EPS_E, EPS_DECAY = 0.10, 0.02, 8000
    BC_LAMBDA = 0.3  # BC regularization weight

    global_step = 0; episode = 0
    best_score = s0
    losses_q, losses_bc = [], []
    round_num = 1
    t0 = time.time()
    checkpoint_path = os.path.join(PROJECT, 'checkpoints', 'phaseB_online_best.pt')
    latest_path = os.path.join(PROJECT, 'checkpoints', 'phaseB_online_latest.pt')

    def should_continue():
        return round_num <= 6  # run 6 rounds (~48k steps) then stop

    while should_continue():
        log(f'\n{"="*45}')
        log(f'ROUND {round_num} | step={global_step} | buf={len(buf)} | {datetime.now().strftime("%H:%M")}')
        log(f'{"="*45}')
        rd_steps = 0; rd_eps = 0; rd_scores = []; rd_deaths = 0

        while rd_steps < 8000 and should_continue():
            eps = EPS_S + (EPS_E - EPS_S) * min(1.0, global_step / EPS_DECAY)
            p = random.choice(GT); gs = mk_ghosts(p)
            s = GameState(); s.initialize(lo, lo.getNumGhosts()); rsh.reset(s)
            pd = None; ep_step = 0

            while not (s.isWin() or s.isLose()) and ep_step < 500:
                g = state_to_grid(s)
                if random.random() < eps:
                    legal = [a for a in s.getLegalActions(0) if a != Directions.STOP] or s.getLegalActions(0)
                    ad = random.choice(legal); mv = ACT[ad]
                else:
                    with torch.no_grad():
                        q = qn(torch.FloatTensor(g).unsqueeze(0).to(dev))[0].cpu().numpy()
                    legal = s.getLegalActions(0)
                    ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
                    if not ids: ids = [4]
                    mv = max(ids, key=lambda i: q[i]); ad = REV[mv]

                # BC teacher action (for regularization)
                teacher_a = dagger_act(g, dagger_m, s)

                s = s.generateSuccessor(0, ad)
                R, gk = rsh.compute(s, ad, pd); pd_died = s.isLose()
                if not (s.isWin() or s.isLose()):
                    for gi, gh in enumerate(gs):
                        if s.isWin() or s.isLose(): break
                        s = s.generateSuccessor(gi+1, gh.getAction(s) or Directions.STOP)
                done = s.isWin() or s.isLose()
                ng = np.zeros((C,H,W), dtype=np.float32) if done else state_to_grid(s)
                if s.isLose() and not pd_died: R -= 500.0
                buf.push(g, mv, R, ng, done, s.isLose(), gk)
                pd = ad; ep_step += 1; global_step += 1; rd_steps += 1

                # Store teacher action for BC loss
                # (We'll compute it during training from the stored state grid)

                # ── Train ──
                if global_step % TRAIN_FREQ == 0 and len(buf) >= BATCH:
                    sb, ab, rb, snb, db = buf.sample(BATCH)
                    st_ = torch.FloatTensor(sb).to(dev)
                    at_ = torch.LongTensor(ab).unsqueeze(1).to(dev)
                    rt_ = torch.FloatTensor(rb).unsqueeze(1).to(dev)
                    snt_ = torch.FloatTensor(snb).to(dev)
                    dt_ = torch.FloatTensor(db).unsqueeze(1).to(dev)

                    # Double DQN loss
                    with torch.no_grad():
                        q_next = tn(snt_)
                        best_a = qn(snt_).argmax(1, keepdim=True)
                        target = rt_ + GAMMA * q_next.gather(1, best_a) * (1 - dt_)
                    q_cur = qn(st_).gather(1, at_)
                    loss_dqn = F.smooth_l1_loss(q_cur, target)

                    # BC regularization: cross-entropy with DAgger teacher
                    with torch.no_grad():
                        # Get teacher action for each state in batch
                        teacher_actions = []
                        for i in range(len(sb)):
                            grid_i = sb[i]
                            t_q = dagger_q(grid_i, dagger_m)
                            # Need legal actions but we don't have state objects here
                            # Use all 5 actions (approximate)
                            teacher_actions.append(t_q.argmax())
                        teacher_a_t = torch.LongTensor(teacher_actions).to(dev)

                    logits = qn(st_)  # Q-values as logits for BC
                    loss_bc = F.cross_entropy(logits, teacher_a_t)

                    loss = loss_dqn + BC_LAMBDA * loss_bc

                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(qn.parameters(), 10.0)
                    opt.step()
                    losses_q.append(loss_dqn.item())
                    losses_bc.append(loss_bc.item())

                if global_step % TGT_UPD == 0:
                    tn.load_state_dict(qn.state_dict())

            rd_eps += 1; episode += 1
            rd_scores.append(s.getScore())
            if s.isLose(): rd_deaths += 1

        if not should_continue(): break

        # ── Eval ──
        avg_s, wr, avg_f, death_r, avg_st = evaluate(qn, dev, 30)
        lq = np.mean(losses_q[-500:]) if losses_q else 0
        lbc = np.mean(losses_bc[-500:]) if losses_bc else 0
        elapsed = (time.time()-t0)/60

        log(f'  Steps={rd_steps} Eps={rd_eps} TrainScore20={np.mean(rd_scores[-20:]):.0f}')
        log(f'  EVAL: score={avg_s:.0f} win={wr:.1%} food%={avg_f:.0f} death={death_r:.1%} steps={avg_st:.0f}')
        log(f'  Loss: dqn={lq:.4f} bc={lbc:.4f} eps={eps:.3f} [{elapsed:.0f}m]')

        # Q-value monitor
        with torch.no_grad():
            test_q = torch.FloatTensor(np.zeros((1,C,H,W), dtype=np.float32)).to(dev)
            q_vals = qn(test_q)[0].cpu().numpy()
        log(f'  Q(zero): [{q_vals[0]:.1f} {q_vals[1]:.1f} {q_vals[2]:.1f} {q_vals[3]:.1f} {q_vals[4]:.1f}]')

        # Check for Q collapse
        if abs(q_vals).max() > 200:
            log(f'  WARNING: Q-values exploding, reloading checkpoint...')
            if os.path.exists(checkpoint_path):
                qn.load_state_dict(torch.load(checkpoint_path, map_location=dev))
                tn.load_state_dict(qn.state_dict())

        if avg_s > best_score:
            best_score = avg_s
            torch.save(qn.state_dict(), checkpoint_path)
            log(f'  [BEST] score={best_score:.0f}')

        torch.save(qn.state_dict(), latest_path)

        # Baseline comparison every 4 rounds
        if round_num % 4 == 0:
            log(f'  --- Baseline comparison ---')
            for nm, md, bi in [('DAgger_R1',dagger_m,False),('DAgger_bias',dagger_m,True),
                               ('Fear_v2',fear2_m,True),('PhaseB_best',[qn],False)]:
                if nm == 'PhaseB_best':
                    s, w = evaluate(qn, dev, 30)[:2]
                else:
                    s, w = eval_baseline(md, bi, 25)
                log(f'    {nm:<16} score={s:7.1f}  win={w:.1%}')

        round_num += 1

    # ── Final ──
    log(f'\n{"="*60}')
    log(f'FINAL @ {datetime.now().strftime("%H:%M")}')
    log(f'{"="*60}')
    if os.path.exists(checkpoint_path):
        qn.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        log(f'Loaded best (score={best_score:.0f})')

    for n_eps in [50, 50]:
        s, w, f, d, st = evaluate(qn, dev, n_eps)
        log(f'  PhaseB DQN ({n_eps}eps): score={s:.0f} win={w:.1%} food%={f:.0f} death={d:.1%} steps={st:.0f}')

    log(f'\n  --- Final comparison (50 eps each) ---')
    for nm, md, bi in [('DAgger_R1',dagger_m,False),('DAgger_bias',dagger_m,True),
                       ('Fear_v2',fear2_m,True),('PhaseB_best',[qn],False)]:
        if nm == 'PhaseB_best':
            s, w = evaluate(qn, dev, 50)[:2]
        else:
            s, w = eval_baseline(md, bi, 50)
        log(f'    {nm:<16} score={s:7.1f}  win={w:.1%}')

    log(f'\nTotal: {global_step} steps, {episode} eps, {(time.time()-t0)/60:.0f}m')


if __name__ == '__main__':
    main()
