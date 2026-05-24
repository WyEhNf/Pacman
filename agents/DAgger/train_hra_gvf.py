"""HRA + GVF — Paper HRA+3 implementation.

GVF (General Value Function):
  - For each food sector center, learn navigation value: Q_gvf_k(s,a) → [0,1]
  - Trained with Expected Sarsa on uniform random policy (stable!)
  - Pseudo-reward: 1 when reaching target cell, else distance-based shaping

Food heads: Q_food_k = GVF_k × 10 (when pellets exist in sector k)
Ghost heads: direct per-head TD (as basic HRA)
Capsule head: GVF to nearest capsule
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
N_GVF = N_FOOD  # one GVF per food sector
N_HEADS = N_FOOD + N_GHOSTS * 2 + 1  # 15
GAMMA = 0.95
N_ACTIONS = 5
BATCH = 256
LR = 1e-4
REPLAY_SIZE = 500000
TARGET_UPDATE = 2000

# ── Food sector centers ──
col_bounds, row_bounds = [0, 4, 8, 12, 16, 20], [0, 6, 11]
FOOD_SECTOR = np.full((H, W), -1, dtype=np.int32)
SECTOR_CENTERS = []  # (y, x) center of each food sector
for ri in range(2):
    for ci in range(5):
        k = ri * 5 + ci
        cx = (col_bounds[ci] + col_bounds[ci + 1]) // 2
        cy = (row_bounds[ri] + row_bounds[ri + 1]) // 2
        SECTOR_CENTERS.append((cy, cx))
        for y in range(row_bounds[ri], row_bounds[ri + 1]):
            for x in range(col_bounds[ci], col_bounds[ci + 1]):
                if 0 <= y < H and 0 <= x < W:
                    FOOD_SECTOR[y, x] = k

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

# ── GVF pseudo-reward: approaching target ──
def gvf_pseudo_reward(grid_prev, grid_curr, target_yx):
    """Dense pseudo-reward for navigation to target (y, x).
    Returns value in [0, 1].
    """
    # Pacman position
    prev_pac = np.argwhere(grid_prev[2] > 0.5)
    curr_pac = np.argwhere(grid_curr[2] > 0.5)

    if len(prev_pac) == 0 or len(curr_pac) == 0:
        return 0.0

    prev_d = abs(prev_pac[0][0] - target_yx[0]) + abs(prev_pac[0][1] - target_yx[1])
    curr_d = abs(curr_pac[0][0] - target_yx[0]) + abs(curr_pac[0][1] - target_yx[1])

    # Reached target
    if curr_d == 0:
        return 1.0

    # Dense progress reward: getting closer → positive, further → negative
    progress = (prev_d - curr_d) / max(prev_d, 1.0)
    return 0.1 + 0.9 * max(0.0, min(1.0, progress + 0.5))  # [0.1, 1.0] range

# ── Sub-reward decomposition (for ghost/capsule heads) ──
def compute_sub_rewards(grid_prev, grid_curr, score_delta):
    sr = np.zeros(N_HEADS, dtype=np.float32)

    # Food: use GVF, not direct reward
    # (food heads will be computed from GVF values, not here)

    # Capsule eaten
    cap_diff = (grid_prev[1] > 0.5) & (grid_curr[1] < 0.5)
    if cap_diff.any():
        sr[-1] += 0.0  # capsules give 0 score

    # Ghost eaten
    for gi in range(N_GHOSTS):
        scared_before = grid_prev[5 + gi].max() > 0.1
        pos_prev = np.argwhere(grid_prev[3 + gi] > 0.5)
        pos_curr = np.argwhere(grid_curr[3 + gi] > 0.5)
        if scared_before and len(pos_prev) > 0 and len(pos_curr) > 0:
            d = abs(pos_prev[0][0] - pos_curr[0][0]) + abs(pos_prev[0][1] - pos_curr[0][1])
            if d > 5:
                sr[N_FOOD + N_GHOSTS + gi] += 200.0

    # Death
    if score_delta < -100:
        pac_pos = np.argwhere(grid_prev[2] > 0.5)
        if len(pac_pos) > 0:
            py0, px0 = pac_pos[0]
            killer, min_d = 0, 999
            for gi in range(N_GHOSTS):
                gp = np.argwhere(grid_prev[3 + gi] > 0.5)
                if len(gp) > 0:
                    d = abs(py0 - gp[0][0]) + abs(px0 - gp[0][1])
                    if d < min_d: min_d = d; killer = gi
            sr[N_FOOD + killer] += score_delta

    return sr

# ── Masks ──
def compute_masks(grid):
    masks = np.zeros(N_HEADS, dtype=np.float32)
    fd = grid[0]
    for y in range(H):
        for x in range(W):
            if fd[y, x] > 0.5:
                s = FOOD_SECTOR[y, x]
                if s >= 0: masks[s] = 1.0
    masks[N_FOOD:N_FOOD + N_GHOSTS] = 1.0
    for gi in range(N_GHOSTS):
        if grid[5 + gi].max() > 0.01:
            masks[N_FOOD + N_GHOSTS + gi] = 1.0
    if grid[1].max() > 0.5:
        masks[-1] = 1.0
    return masks

# ── HRA+GVF Model ──
class HRA_GVF_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared visual backbone
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.shared_fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU())

        # Navigation GVFs — one per food sector
        # Each GVF outputs Q_gvf(s,a) for navigating to its sector center
        self.gvfs = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, N_ACTIONS))
            for _ in range(N_GVF)
        ])

        # Ghost + capsule heads (non-GVF, direct TD)
        self.non_gvf_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, N_ACTIONS))
            for _ in range(N_GHOSTS * 2 + 1)  # danger×2 + prey×2 + capsule
        ])

    def forward(self, x, masks=None):
        B = x.size(0)
        f = self.shared_fc(self.conv(x).mean(dim=[2, 3]))

        # GVF outputs (food navigation)
        gvf_qs = torch.stack([g(f) for g in self.gvfs], dim=1)  # (B, N_GVF, 5)

        # Non-GVF outputs (ghost danger/prey, capsule)
        other_qs = torch.stack([h(f) for h in self.non_gvf_heads], dim=1)  # (B, N_GHOSTS*2+1, 5)

        # Combine: [food_gvfs | ghost_heads | capsule_head]
        q_heads = torch.cat([gvf_qs, other_qs], dim=1)  # (B, N_HEADS, 5)

        if masks is not None:
            q_total = (q_heads * masks.to(x.device).unsqueeze(-1)).sum(dim=1)
        else:
            q_total = q_heads.sum(dim=1)
        return q_total, q_heads, gvf_qs

# ── Replay Buffer ──
Transition = collections.namedtuple if False else type('Transition', (), {})

class ReplayBuffer:
    def __init__(self, capacity): self.buf = deque(maxlen=capacity)
    def add(self, *args): self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        return [torch.FloatTensor(np.stack([b[i] for b in batch])) for i in range(len(batch[0]))]
    def __len__(self): return len(self.buf)

# ── Fill replay buffer ──
def fill_replay():
    buf = ReplayBuffer(REPLAY_SIZE)

    def process_traj(grids, actions):
        T = len(actions)
        for i in range(T - 1):
            s, ns = grids[i], grids[i + 1]
            a = int(actions[i])

            # GVF pseudo-rewards
            gvf_rewards = np.zeros(N_GVF, dtype=np.float32)
            for k in range(N_GVF):
                gvf_rewards[k] = gvf_pseudo_reward(s, ns, SECTOR_CENTERS[k])

            # Non-GVF sub-rewards
            fd_diff = (s[0] > 0.5) & (ns[0] < 0.5)
            score_delta = 10.0 * fd_diff.sum()
            if ns[2].max() < 0.1: score_delta = -500.0
            sub_r = compute_sub_rewards(s, ns, score_delta)

            done = float(ns[2].max() < 0.1)
            m, nm = compute_masks(s), compute_masks(ns)

            buf.add(s, a, ns, done, gvf_rewards, sub_r, m, nm)
            # Transitions stored as: 0:s, 1:a, 2:ns, 3:done, 4:gvf_r, 5:sub_r, 6:m, 7:nm

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

    for fname in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fname)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                process_traj(t['states'], t['actions'])

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
                q_total, _, _ = model(t, mt)
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
    print(f'HRA + GVF: {N_GVF} navigation GVFs + {N_GHOSTS*2+1} direct heads')

    buf = fill_replay()

    online = HRA_GVF_Model().to(device)
    target = HRA_GVF_Model().to(device)
    target.load_state_dict(online.state_dict())

    # Warm-start conv
    warm_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    if os.path.exists(warm_path):
        pretrained = torch.load(warm_path, map_location=device)
        conv_keys = {k: v for k, v in pretrained.items() if 'conv.' in k}
        if conv_keys:
            online.conv.load_state_dict({k.replace('conv.', ''): v for k, v in conv_keys.items()})
            print(f'Warm-started conv')

    opt = torch.optim.Adam(online.parameters(), lr=LR)
    base_avg, base_wins = eval_model(online, device, 10)
    print(f'Baseline: avg={base_avg:.0f} wins={base_wins}/10')

    total_steps = 0
    best_avg = base_avg

    for epoch in range(200):
        online.train()
        losses_gvf, losses_other = [], []
        n_batches = min(400, len(buf) // BATCH)

        for _ in range(n_batches):
            s, a, ns, done, gvf_r, sub_r, m, nm = buf.sample(BATCH)
            s = s.to(device); a = a.to(device).long(); ns = ns.to(device)
            done = done.to(device); gvf_r = gvf_r.to(device)
            sub_r = sub_r.to(device); m = m.to(device); nm = nm.to(device)

            # Current Q
            _, q_heads, gvf_qs = online(s, m)
            N_NON_GVF = q_heads.size(1) - N_GVF
            B = s.size(0)

            # ── Compute TD targets ──
            with torch.no_grad():
                _, tgt_heads, tgt_gvfs = target(ns, nm)
                # GVF target (Expected Sarsa: average over uniform policy)
                gvf_target = gvf_r + GAMMA * tgt_gvfs.mean(dim=-1) * (1 - done.unsqueeze(-1))
                gvf_target = torch.clamp(gvf_target, 0.0, 2.0)
                # Non-GVF target (standard max-Q)
                tgt_non_gvf = tgt_heads[:, N_GVF:]
                max_q_next = tgt_non_gvf.max(dim=-1).values
                if sub_r.size(1) <= N_GVF:
                    sub_r_ngvf = torch.zeros(B, N_NON_GVF, device=device)
                else:
                    sub_r_ngvf = sub_r[:, N_GVF:N_GVF+N_NON_GVF]
                td_target = sub_r_ngvf + GAMMA * max_q_next * (1 - done.unsqueeze(-1))
                td_target = torch.clamp(td_target, -600, 600)

            # ── Compute losses ──
            gvf_acted = gvf_qs.gather(2, a.unsqueeze(1).unsqueeze(2).expand(-1, N_GVF, 1)).squeeze(-1)
            loss_gvf = F.mse_loss(gvf_acted, gvf_target)

            non_gvf_qs = q_heads[:, N_GVF:]
            non_gvf_acted = non_gvf_qs.gather(2, a.unsqueeze(1).unsqueeze(2).expand(-1, N_NON_GVF, 1)).squeeze(-1)
            loss_other = F.huber_loss(non_gvf_acted, td_target, delta=20.0)

            loss = loss_gvf + loss_other
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            opt.step()

            losses_gvf.append(loss_gvf.item())
            losses_other.append(loss_other.item())
            total_steps += 1
            if total_steps % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict())

        print(f'E{epoch:3d} | GVF={np.mean(losses_gvf):.4f} Other={np.mean(losses_other):.4f}')

        if (epoch + 1) % 20 == 0:
            avg, wins = eval_model(online, device, 10)
            print(f'  Eval: avg={avg:.0f} wins={wins}/10')
            if avg > best_avg:
                best_avg = avg
                torch.save(online.state_dict(), os.path.join(PROJECT, 'checkpoints', 'hra_gvf_best.pt'))

    final_avg, final_wins = eval_model(online, device, 20)
    print(f'\nFinal: avg={final_avg:.0f} wins={final_wins}/20')
    torch.save(online.state_dict(), os.path.join(PROJECT, 'checkpoints', 'hra_gvf_final.pt'))

if __name__ == '__main__':
    import collections
    main()
