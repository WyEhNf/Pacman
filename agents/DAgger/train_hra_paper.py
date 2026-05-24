"""HRA — Strict paper implementation (NIPS 2017).

Per-head TD learning + masked activation + sub-reward decomposition.
Training: each head k minimizes TD error on its own reward component R_k.
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

H, W, C = 11, 20, 8
N_FOOD = 10
N_GHOSTS = 2
N_HEADS = N_FOOD + N_GHOSTS * 2 + 1  # 15
GAMMA = 0.95
BATCH = 256
LR = 1e-4
TARGET_UPDATE = 2000
REPLAY_SIZE = 500000

# ── Food sectors ──
col_bounds, row_bounds = [0, 4, 8, 12, 16, 20], [0, 6, 11]
FOOD_SECTOR = np.full((H, W), -1, dtype=np.int32)
for y in range(H):
    for x in range(W):
        ci = next(c for c in range(len(col_bounds)-1) if col_bounds[c] <= x < col_bounds[c+1])
        ri = next(r for r in range(len(row_bounds)-1) if row_bounds[r] <= y < row_bounds[r+1])
        FOOD_SECTOR[y, x] = ri * 5 + ci

# ── Walls ──
def get_walls_grid():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo

WG, LO = get_walls_grid()

# ── State → Grid ──
def flat_to_grid(feat):
    g = np.zeros((C, H, W), dtype=np.float32)
    g[0] = feat[8:8+H*W].reshape(W, H).T
    g[1] = feat[228:228+H*W].reshape(W, H).T
    px = min(max(int(feat[0] * H), 0), W - 1)
    py = min(max(int(feat[1] * W), 0), H - 1)
    g[2, py, px] = 1.0
    for i in range(N_GHOSTS):
        gx = min(max(int(feat[2 + i*2] * H), 0), W - 1)
        gy = min(max(int(feat[3 + i*2] * W), 0), H - 1)
        g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = feat[6 + i]
    g[7] = WG; return g

def state_to_grid(state):
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
    g[7] = WG; return g

# ── Sub-reward decomposition ──
def compute_sub_rewards(grid_prev, grid_curr, score_delta):
    """Decompose score_delta into N_HEADS sub-rewards.
    Detects: food eaten, capsules eaten, ghost eaten, pacman death.
    """
    sr = np.zeros(N_HEADS, dtype=np.float32)

    # Food eaten (channel 0, 1→0 transitions)
    fd_prev, fd_curr = grid_prev[0], grid_curr[0]
    for y in range(H):
        for x in range(W):
            if fd_prev[y, x] > 0.5 and fd_curr[y, x] < 0.5:
                s = FOOD_SECTOR[y, x]
                if s >= 0:
                    sr[s] += 10.0

    # Capsule eaten (channel 1, 1→0)
    cap_prev, cap_curr = grid_prev[1], grid_curr[1]
    cap_eaten = (cap_prev > 0.5) & (cap_curr < 0.5)
    if cap_eaten.any():
        sr[-1] += 0.0  # capsules give 0 score but the event matters

    # Ghost eaten (ghost was scared, position jumped >5 cells)
    for gi in range(N_GHOSTS):
        scared_before = grid_prev[5 + gi].max() > 0.1
        scared_after = grid_curr[5 + gi].max() > 0.1
        pos_prev = np.argwhere(grid_prev[3 + gi] > 0.5)
        pos_curr = np.argwhere(grid_curr[3 + gi] > 0.5)
        if scared_before and len(pos_prev) > 0 and len(pos_curr) > 0:
            d = abs(pos_prev[0][0] - pos_curr[0][0]) + abs(pos_prev[0][1] - pos_curr[0][1])
            if d > 5:  # ghost respawned
                sr[N_FOOD + N_GHOSTS + gi] += 200.0

    # Pacman death (large negative score)
    if score_delta < -100:
        # Find nearest non-scared ghost to pacman
        pac_pos = np.argwhere(grid_prev[2] > 0.5)
        if len(pac_pos) > 0:
            py0, px0 = pac_pos[0]
            min_dist = 999
            killer = 0
            for gi in range(N_GHOSTS):
                ghost_pos = np.argwhere(grid_prev[3 + gi] > 0.5)
                if len(ghost_pos) > 0:
                    d = abs(py0 - ghost_pos[0][0]) + abs(px0 - ghost_pos[0][1])
                    if d < min_dist: min_dist = d; killer = gi
            sr[N_FOOD + killer] += score_delta  # danger head

    # Normalize: ensure sum matches score_delta
    current_sum = sr.sum()
    if abs(current_sum - score_delta) > 0.1 and abs(current_sum) > 0.01:
        # Scale to match
        if current_sum != 0:
            sr = sr * (score_delta / current_sum)

    return sr

# ── Masks ──
def compute_masks(grid):
    masks = np.zeros(N_HEADS, dtype=np.float32)
    # Food heads
    fd = grid[0]
    for y in range(H):
        for x in range(W):
            if fd[y, x] > 0.5:
                s = FOOD_SECTOR[y, x]
                if s >= 0: masks[s] = 1.0
    # Ghost danger always active
    masks[N_FOOD:N_FOOD + N_GHOSTS] = 1.0
    # Ghost prey
    for gi in range(N_GHOSTS):
        if grid[5 + gi].max() > 0.01:
            masks[N_FOOD + N_GHOSTS + gi] = 1.0
    # Capsule
    if grid[1].max() > 0.5:
        masks[-1] = 1.0
    return masks

# ── HRA Model ──
class HRA_DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.shared_fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 5))
            for _ in range(N_HEADS)
        ])

    def forward(self, x, masks=None):
        B = x.size(0)
        f = self.shared_fc(self.conv(x).mean(dim=[2, 3]))
        q_heads = torch.stack([h(f) for h in self.heads], dim=1)  # (B, N_HEADS, 5)
        if masks is not None:
            q_total = (q_heads * masks.to(x.device).unsqueeze(-1)).sum(dim=1)
        else:
            q_total = q_heads.sum(dim=1)
        return q_total, q_heads

# ── Replay Buffer ──
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, s_next, done, sub_rewards, masks, next_masks):
        self.buf.append((s, a, s_next, done, sub_rewards, masks, next_masks))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, ns = [], [], []
        done, sr, m, nm = [], [], [], []
        for item in batch:
            s.append(item[0]); a.append(item[1]); ns.append(item[2])
            done.append(item[3]); sr.append(item[4])
            m.append(item[5]); nm.append(item[6])
        return (torch.FloatTensor(np.stack(s)),
                torch.LongTensor(np.array(a)),
                torch.FloatTensor(np.stack(ns)),
                torch.FloatTensor(np.array(done, dtype=np.float32)),
                torch.FloatTensor(np.stack(sr)),
                torch.FloatTensor(np.stack(m)),
                torch.FloatTensor(np.stack(nm)))

    def __len__(self): return len(self.buf)

# ── Load data into replay buffer ──
def fill_replay():
    buf = ReplayBuffer(REPLAY_SIZE)
    act_map = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
               Directions.WEST: 3, Directions.STOP: 4}

    def process_traj(grids, actions):
        T = len(actions)
        for i in range(T - 1):
            s = grids[i]; ns = grids[i + 1]
            a = int(actions[i])
            # Score delta estimation: detect food changes
            food_diff = (s[0] > 0.5) & (ns[0] < 0.5)
            ghost_eaten = False
            for gi in range(N_GHOSTS):
                if s[5+gi].max() > 0.1 and abs(s[3+gi].argmax() - ns[3+gi].argmax()) > 1000:
                    pass  # rough check
            raw_delta = 10.0 * food_diff.sum()
            # Detect death: pacman disappears from ns
            if ns[2].max() < 0.1:  # pacman not in next frame = died
                raw_delta = -500.0
            sr = compute_sub_rewards(s, ns, raw_delta)
            done = float(ns[2].max() < 0.1)
            m = compute_masks(s)
            nm = compute_masks(ns)
            buf.add(s, a, ns, done, sr, m, nm)

    # Original data
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']
        T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        process_traj(grids, a)

    # DAgger data
    for fname in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fname)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                process_traj(t['states'], t['actions'])

    # Self-play data (last 3)
    for fp in sorted(glob.glob(os.path.join(PROJECT, 'data', 'selfplay_r*.npz')))[-3:]:
        d = np.load(fp, allow_pickle=True)
        for t in d['trajectories']:
            process_traj(t['states'], t['actions'])

    print(f'Replay buffer: {len(buf)} transitions')
    return buf

# ── Eval ──
def make_ghosts(profile='balanced'):
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(LO.getNumGhosts())]
    att = {'aggressive': 0.9, 'balanced': 0.5, 'coward': 0.2}[profile]
    fle = {'aggressive': 0.2, 'balanced': 0.5, 'coward': 0.9}[profile]
    return [ghostAgents.DirectionalGhost(i + 1, att, fle) for i in range(LO.getNumGhosts())]

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

def eval_model(model, device, n_eps=10):
    model.eval()
    scores, wins = [], 0
    for _ in range(n_eps):
        st = GameState(); st.initialize(LO, LO.getNumGhosts())
        ghosts = make_ghosts('balanced')
        step = 0
        while not (st.isWin() or st.isLose()) and step < 500:
            g = state_to_grid(st)
            masks = compute_masks(g)
            with torch.no_grad():
                t = torch.FloatTensor(g).unsqueeze(0).to(device)
                mt = torch.FloatTensor(masks).unsqueeze(0).to(device)
                q_total, _ = model(t, mt)
                q = q_total[0].cpu().numpy()
            legal = st.getLegalActions(0)
            ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
            if not ids: ids = [4]
            best, mv = -1e9, 4
            for i in range(5):
                if i in ids and q[i] > best: best = q[i]; mv = i
            st = st.generateSuccessor(0, REV[mv])
            if st.isWin() or st.isLose(): break
            for gi, g in enumerate(ghosts):
                if st.isWin() or st.isLose(): break
                st = st.generateSuccessor(gi + 1, g.getAction(st) or Directions.STOP)
            step += 1
        scores.append(st.getScore())
        if st.isWin(): wins += 1
    return np.mean(scores), wins

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'HRA Paper: {N_HEADS} heads, per-head TD with replay buffer')

    buf = fill_replay()

    # Online + Target networks
    online = HRA_DQN().to(device)
    target = HRA_DQN().to(device)
    target.load_state_dict(online.state_dict())

    # Warm-start CNN from best pretrained model
    warm_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    if os.path.exists(warm_path):
        pretrained = torch.load(warm_path, map_location=device)
        # Load conv + fc weights (skip head weights since architecture differs)
        online.conv.load_state_dict({k.replace('conv.', ''): v for k, v in pretrained.items() if 'conv.' in k})
        print(f'Warm-started conv from {warm_path}')

    opt = torch.optim.Adam(online.parameters(), lr=LR)

    # Baseline eval
    base_avg, base_wins = eval_model(online, device, 10)
    print(f'Baseline: avg={base_avg:.0f} wins={base_wins}/10')

    # Training
    total_steps = 0
    best_avg = base_avg
    log = open(os.path.join(PROJECT, 'hra_paper_log.txt'), 'w')

    for epoch in range(200):
        online.train()
        td_losses, q_means = [], [[] for _ in range(N_HEADS)]

        batches_per_epoch = min(400, len(buf) // BATCH)
        for _ in range(batches_per_epoch):
            s, a, ns, done, sr, m, nm = buf.sample(BATCH)
            s, a = s.to(device), a.to(device)
            ns, done = ns.to(device), done.to(device)
            sr, m, nm = sr.to(device), m.to(device), nm.to(device)

            # Current Q per head
            _, q_heads = online(s, m)          # (B, N_HEADS, 5)
            q_acted = q_heads.gather(2, a.unsqueeze(1).unsqueeze(2).expand(-1, N_HEADS, 1))
            q_acted = q_acted.squeeze(-1)       # (B, N_HEADS)

            # Target Q per head
            with torch.no_grad():
                _, tgt_heads = target(ns, nm)  # (B, N_HEADS, 5)
                max_q_next = tgt_heads.max(dim=-1).values  # (B, N_HEADS)
                td_target = sr + GAMMA * max_q_next * (1 - done.unsqueeze(-1))
                td_target = torch.clamp(td_target, -600, 600)

            # Per-head TD loss
            td_loss = F.huber_loss(q_acted, td_target, delta=20.0)
            opt.zero_grad()
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            opt.step()

            td_losses.append(td_loss.item())
            for k in range(N_HEADS):
                q_means[k].append(q_acted[:, k].mean().item())

            total_steps += 1
            if total_steps % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict())

        # Logging
        td_mean = np.mean(td_losses)
        q_head_means = [np.mean(qm) for qm in q_means]
        print(f'E{epoch:3d} | TD_loss={td_mean:.4f} | '
              f'Q_food={np.mean(q_head_means[:N_FOOD]):.1f} '
              f'Q_danger={np.mean(q_head_means[N_FOOD:N_FOOD+N_GHOSTS]):.1f} '
              f'Q_prey={np.mean(q_head_means[N_FOOD+N_GHOSTS:N_FOOD+N_GHOSTS*2]):.1f}')

        log.write(f'E{epoch}: TD={td_mean:.4f}\n')

        # Eval every 20 epochs
        if (epoch + 1) % 20 == 0:
            avg, wins = eval_model(online, device, 10)
            print(f'  Eval: avg={avg:.0f} wins={wins}/10')
            log.write(f'  Eval: avg={avg:.0f} wins={wins}/10\n')
            if avg > best_avg:
                best_avg = avg
                torch.save(online.state_dict(), os.path.join(PROJECT, 'checkpoints', 'hra_paper_best.pt'))

        log.flush()

    final_avg, final_wins = eval_model(online, device, 20)
    print(f'\nFinal: avg={final_avg:.0f} wins={final_wins}/20')
    torch.save(online.state_dict(), os.path.join(PROJECT, 'checkpoints', 'hra_paper_final.pt'))
    log.close()

if __name__ == '__main__':
    main()
