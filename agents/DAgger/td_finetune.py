"""TD Fine-Tuning — CE anchor + tiny TD loss to push past imitation ceiling.

Safety measures (all lessons from previous TD explosion):
  - λ_td = 0.001 (1000x smaller than CE)
  - Target network with soft update (Polyak τ=0.005)
  - Huber loss (robust to outliers)
  - Gradient clipping at 10.0
  - Q target clamped to [-500, 1000]
  - TD only applied to self-play transitions (not expert)
  - Stop signal: if Q values exceed ±500 for 3 consecutive batches, abort

Usage:
  python scripts/td_finetune.py --rounds 10 --steps 500 --eval-every 2
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

H, W, C = 11, 20, 8
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

parser = argparse.ArgumentParser()
parser.add_argument('--td-weight', type=float, default=0.001)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--steps-per-round', type=int, default=500)
parser.add_argument('--eval-every', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--tau', type=float, default=0.005)
args = parser.parse_args()

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

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=[2, 3]))

# ── Grid utils ──
def get_walls_grid():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo

WG, LO = get_walls_grid()

def flat_to_grid(feat):
    g = np.zeros((C, H, W), dtype=np.float32)
    g[0] = feat[8:8+H*W].reshape(W, H).T
    g[1] = feat[228:228+H*W].reshape(W, H).T
    px = min(max(int(feat[0] * H), 0), W - 1)
    py = min(max(int(feat[1] * W), 0), H - 1)
    g[2, py, px] = 1.0
    for i in range(2):
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

# ── Load data ──
def load_data():
    """Returns (expert_trajs, selfplay_trajs_with_rewards)."""
    expert, sp = [], []

    # Original data (expert, no TD)
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']
        T = len(a); r = t.get('rewards', np.zeros(T))
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        expert.append({'grids': grids, 'actions': a, 'rewards': r})

    # Self-play data (has rewards from game engine)
    import glob
    for fp in sorted(glob.glob(os.path.join(PROJECT, 'data', 'selfplay_r*.npz'))):
        d = np.load(fp, allow_pickle=True)
        for t in d['trajectories']:
            s, a = t['states'], t['actions']
            T = len(a)
            if T < 2: continue
            r = np.zeros(T, dtype=np.float32)
            # Reconstruct rewards from score differences
            if 'score' in t and 'steps' in t:
                N = t['steps']
                rr = t.get('rewards', None)
                if rr is not None and len(rr) == N:
                    r[:N] = rr[:N]
            sp.append({'grids': s, 'actions': a, 'rewards': r})

    print(f'Expert data: {len(expert)} eps')
    print(f'Self-play data: {len(sp)} eps')
    return expert, sp

# ── TD Dataset (from self-play trajectories with rewards) ──
class TD_Dataset(Dataset):
    """Creates (s, a, r, s', done) transitions from self-play trajectories."""
    def __init__(self, trajs):
        self.transitions = []
        for t in trajs:
            g = t['grids']
            a = t['actions']
            r = t['rewards']
            T = len(a)
            for i in range(T - 1):
                self.transitions.append((
                    g[i], int(a[i]), float(r[i]) if i < len(r) else 0.0,
                    g[i + 1], False
                ))
            # Last transition
            if T >= 1:
                self.transitions.append((
                    g[T - 1], int(a[T - 1]),
                    float(r[T - 1]) if T - 1 < len(r) else 0.0,
                    g[T - 1], True  # terminal
                ))

    def __len__(self): return len(self.transitions)
    def __getitem__(self, idx):
        s, a, r, ns, d = self.transitions[idx]
        return (torch.FloatTensor(s), torch.LongTensor([a]),
                torch.FloatTensor([r]), torch.FloatTensor(ns),
                torch.FloatTensor([float(d)]))

# ── CE Dataset ──
class CE_Dataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            g, a = t['grids'], t['actions']
            for i in range(len(a) - 1):
                self.samples.append((g[i], int(a[i + 1])))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, a = self.samples[idx]
        return torch.FloatTensor(g), torch.LongTensor([a])

# ── Agent (for eval) ──
def make_ghosts(profile='balanced'):
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(LO.getNumGhosts())]
    att = {'aggressive': 0.9, 'balanced': 0.5, 'coward': 0.2}[profile]
    fle = {'aggressive': 0.2, 'balanced': 0.5, 'coward': 0.9}[profile]
    return [ghostAgents.DirectionalGhost(i + 1, att, fle) for i in range(LO.getNumGhosts())]

def eval_model(model, device, n_eps=10):
    model.eval()
    scores, wins = [], 0
    ghosts = make_ghosts('balanced')
    for _ in range(n_eps):
        st = GameState(); st.initialize(LO, LO.getNumGhosts())
        step = 0
        while not (st.isWin() or st.isLose()) and step < 500:
            with torch.no_grad():
                t = torch.FloatTensor(state_to_grid(st)).unsqueeze(0).to(device)
                q = model(t)[0].cpu().numpy()
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
    print(f'Config: td_weight={args.td_weight}, lr={args.lr}, '
          f'gamma={args.gamma}, tau={args.tau}')
    print(f'Rounds: {args.rounds}, steps/round: {args.steps_per_round}')

    expert_trajs, sp_trajs = load_data()
    ce_ds = CE_Dataset(expert_trajs + sp_trajs)
    ce_loader = DataLoader(ce_ds, 256, shuffle=True, drop_last=True)
    td_ds = TD_Dataset(sp_trajs)
    td_loader = DataLoader(td_ds, 256, shuffle=True, drop_last=True)
    print(f'CE samples: {len(ce_ds)}  TD samples: {len(td_ds)}')

    # Load starting model (best self-play checkpoint)
    model = CNNDQN().to(device)
    target = CNNDQN().to(device)
    model_path = os.path.join(PROJECT, 'checkpoints', 'selfplay_latest_m0.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT, 'checkpoints', 'sp_r4_m0.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    target.load_state_dict(model.state_dict())
    print(f'Loaded: {model_path}')

    # Baseline eval
    base_avg, base_wins = eval_model(model, device, 20)
    print(f'Baseline: avg={base_avg:.0f} wins={base_wins}/20')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_iter = iter(ce_loader)
    td_iter = iter(td_loader)
    log = open(os.path.join(PROJECT, 'td_finetune_log.txt'), 'w')

    explosion_warn = 0  # consecutive explosion checks

    for rnd in range(args.rounds):
        print(f'\n--- Round {rnd + 1}/{args.rounds} ---')
        model.train()
        rnd_ce, rnd_td, rnd_q = [], [], []

        for step in range(args.steps_per_round):
            # Get CE batch
            try:
                grids_ce, acts_ce = next(ce_iter)
            except StopIteration:
                ce_iter = iter(ce_loader)
                grids_ce, acts_ce = next(ce_iter)

            # Get TD batch
            try:
                s_td, a_td, r_td, ns_td, d_td = next(td_iter)
            except StopIteration:
                td_iter = iter(td_loader)
                s_td, a_td, r_td, ns_td, d_td = next(td_iter)

            grids_ce = grids_ce.to(device)
            acts_ce = acts_ce.to(device)
            s_td, a_td = s_td.to(device), a_td.to(device)
            r_td, ns_td, d_td = r_td.to(device), ns_td.to(device), d_td.to(device)

            # ── CE Loss (anchor) ──
            q_ce = model(grids_ce)
            ce_loss = F.cross_entropy(q_ce, acts_ce.squeeze(-1))

            # ── TD Loss (push beyond imitation) ──
            q_td = model(s_td).gather(1, a_td).squeeze()  # Q(s, a)
            with torch.no_grad():
                q_next = target(ns_td).max(dim=-1).values  # max_a' Q_target(s', a')
                td_target = r_td.squeeze() + args.gamma * q_next * (1 - d_td.squeeze())
                td_target = torch.clamp(td_target, -500, 1000)

            td_loss = F.huber_loss(q_td, td_target, delta=10.0)

            # ── Combined ──
            loss = ce_loss + args.td_weight * td_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            # Soft update target
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.data = args.tau * p.data + (1 - args.tau) * tp.data

            rnd_ce.append(ce_loss.item())
            rnd_td.append(td_loss.item())
            rnd_q.append(q_td.mean().item())

            # ── Safety: Q-value explosion detection ──
            q_mean = np.mean(rnd_q[-50:]) if len(rnd_q) >= 50 else np.mean(rnd_q)
            if abs(q_mean) > 500:
                explosion_warn += 1
                if explosion_warn >= 3:
                    print(f'FATAL: Q-value explosion detected (mean={q_mean:.0f}). Aborting.')
                    torch.save(model.state_dict(),
                               os.path.join(PROJECT, 'checkpoints', 'td_ABORTED.pt'))
                    log.close()
                    return
            else:
                explosion_warn = 0

        # ── Eval ──
        ce_m, td_m, q_m = np.mean(rnd_ce), np.mean(rnd_td), np.mean(rnd_q)
        log_msg = f'R{rnd+1}: CE={ce_m:.4f} TD={td_m:.2f} Qmean={q_m:.1f}'
        print(f'  {log_msg}', end='')
        log.write(log_msg)

        if (rnd + 1) % args.eval_every == 0:
            avg, wins = eval_model(model, device, 10)
            eval_msg = f' | Eval: avg={avg:.0f} wins={wins}/10'
            print(eval_msg)
            log.write(eval_msg)

            ckpt = os.path.join(PROJECT, 'checkpoints', f'td_finetune_r{rnd+1}.pt')
            torch.save(model.state_dict(), ckpt)
        else:
            print()
            log.write('\n')

        log.flush()

    # Final eval
    final_avg, final_wins = eval_model(model, device, 20)
    print(f'\nFinal: avg={final_avg:.0f} wins={final_wins}/20')
    print(f'Baseline was: avg={base_avg:.0f} wins={base_wins}/20')
    torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints', 'td_finetune_final.pt'))
    log.write(f'\nFinal: avg={final_avg:.0f} wins={final_wins}/20\n')
    log.close()

if __name__ == '__main__':
    main()
