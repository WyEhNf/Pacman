"""Continuous Self-Play Pipeline.

Model plays → saves trajectories → retrains → plays better → repeats forever.

Architecture:
  - Start with DAgger R1 Ensemble as base model
  - Self-play: ensemble plays episodes, records ALL trajectories
  - Data pool: original + DAgger + rolling window of recent self-play
  - Retrain: every SELFPLAY_EPS_PER_ROUND episodes, train 3-model CNN ensemble
  - Model update: new ensemble replaces old for next round
"""
import sys, os, time, signal, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

# ── Config ──
H, W, C = 11, 20, 8
N_ENSEMBLE = 3          # models per ensemble (faster retrain)
SELFPLAY_EPS_PER_ROUND = 100   # play 100 eps before retraining
SELFPLAY_BUFFER_SIZE = 300     # rolling window of recent self-play eps
BATCH = 128
RETRAIN_EPOCHS = 100
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

GHOST_PROFILES = ['aggressive', 'balanced', 'random', 'coward']

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

# ── Grid conversion ──
def get_walls_grid():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo

WG, LO = get_walls_grid()

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

def make_ghosts(profile):
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(LO.getNumGhosts())]
    a = {'aggressive': 0.9, 'balanced': 0.5, 'coward': 0.2}[profile]
    f = {'aggressive': 0.2, 'balanced': 0.5, 'coward': 0.9}[profile]
    return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(LO.getNumGhosts())]

# ── Ensemble Agent ──
class EnsembleAgent:
    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device

    def getAction(self, state):
        g = state_to_grid(state)
        with torch.no_grad():
            t = torch.FloatTensor(g).unsqueeze(0).to(self.device)
            q = sum(m(t)[0].cpu().numpy() for m in self.models) / len(self.models)
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        return REV[mv]

# ── Run one episode ──
def run_episode(agent, ghost_profile):
    state = GameState(); state.initialize(LO, LO.getNumGhosts())
    ghosts = make_ghosts(ghost_profile)
    grids, actions, rewards = [], [], []
    prev_score = state.getScore()
    step = 0

    while not (state.isWin() or state.isLose()) and step < 500:
        action = agent.getAction(state)
        grids.append(state_to_grid(state))
        actions.append(ACT[action])
        state = state.generateSuccessor(0, action)
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi + 1, g.getAction(state) or Directions.STOP)
        rewards.append(state.getScore() - prev_score)
        prev_score = state.getScore()
        step += 1

    if len(grids) < 5: return None
    return {'states': np.array(grids, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int32),
            'score': state.getScore(), 'win': state.isWin(),
            'steps': len(grids), 'ghost': ghost_profile}

# ── Load base data (original + DAgger) ──
def load_base_data():
    """Load original + DAgger data as grid-native trajectories."""
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

    trajs = []
    # Original 470 eps
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']
        T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        trajs.append({'states': grids, 'actions': a})
    print(f'  Base original: {len(trajs)} eps')

    # DAgger data
    for fname in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fname)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                trajs.append({'states': t['states'], 'actions': t['actions']})
            print(f'  +{fname}: {len(d["trajectories"])} eps')
    return trajs

# ── Dataset for retraining ──
class RetrainDataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s, a = t['states'], t['actions']
            T = len(a)
            if T < 2: continue
            for i in range(T - 1):
                self.samples.append((s[i], int(a[i + 1])))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, a = self.samples[idx]
        return torch.FloatTensor(g), torch.LongTensor([a])

# ── Train ensemble ──
def train_ensemble(train_trajs, device, round_num):
    """Train N_ENSEMBLE CNN models. Returns list of models."""
    np.random.seed(42); np.random.shuffle(train_trajs)
    split = int(len(train_trajs) * 0.9)
    train_ds = RetrainDataset(train_trajs[:split])
    val_ds = RetrainDataset(train_trajs[split:])
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}')

    models = []
    for mid in range(N_ENSEMBLE):
        torch.manual_seed(mid); np.random.seed(mid)
        model = CNNDQN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_val = 0.0

        for ep in range(RETRAIN_EPOCHS):
            model.train()
            for grids, actions in train_loader:
                grids, actions = grids.to(device), actions.to(device)
                q = model(grids)
                loss = F.cross_entropy(q, actions.squeeze(-1))
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            v_acc = []
            with torch.no_grad():
                for grids, actions in val_loader:
                    grids, actions = grids.to(device), actions.to(device)
                    q = model(grids)
                    v_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())
            val_acc = np.mean(v_acc)
            if val_acc > best_val:
                best_val = val_acc
                ckpt = os.path.join(PROJECT, 'checkpoints', f'sp_r{round_num}_m{mid}.pt')
                torch.save(model.state_dict(), ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval(); models.append(model)
        print(f'  M{mid}: val_acc={best_val:.3f}')

    return models

# ── Main Loop ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: {N_ENSEMBLE} models, {SELFPLAY_EPS_PER_ROUND} eps/round, '
          f'buffer={SELFPLAY_BUFFER_SIZE}')

    # Load base data
    base_trajs = load_base_data()
    print(f'Base data: {len(base_trajs)} eps total')

    # Start with DAgger R1 ensemble
    ensemble_models = []
    for i in range(N_ENSEMBLE):
        m = CNNDQN().to(device)
        m.load_state_dict(torch.load(
            os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location=device))
        m.eval(); ensemble_models.append(m)
    agent = EnsembleAgent(ensemble_models, device)
    print(f'Initial model: DAgger R1 Ensemble ({len(ensemble_models)} models)')

    selfplay_buffer = []  # rolling window of recent self-play eps
    round_num = 1
    total_sp_eps = 0
    total_wins = 0
    stopped = False

    def on_sig(sig, frame):
        nonlocal stopped; stopped = True; print('\nStopping gracefully...')
    signal.signal(signal.SIGINT, on_sig)

    log_path = os.path.join(PROJECT, 'selfplay_log.txt')
    log = open(log_path, 'a')
    log.write(f'\n=== Self-Play Started {datetime.now()} ===\n')
    log.write(f'Base data: {len(base_trajs)} eps, Initial: DAgger R1\n')

    while not stopped:
        print(f'\n{"="*60}')
        print(f'  Round {round_num} | Self-Play: {SELFPLAY_EPS_PER_ROUND} eps')
        print(f'  Buffer: {len(selfplay_buffer)} sp eps | '
              f'Total: {total_sp_eps} sp / {total_wins} wins')
        print(f'{"="*60}')

        # ── Self-Play Collection ──
        round_scores, round_wins = [], 0
        t0 = time.time()
        for i in range(SELFPLAY_EPS_PER_ROUND):
            ghost = np.random.choice(GHOST_PROFILES)
            traj = run_episode(agent, ghost)
            if traj is None: continue
            selfplay_buffer.append(traj)
            round_scores.append(traj['score'])
            if traj['win']: round_wins += 1
            if (i + 1) % 20 == 0:
                recent = round_scores[-20:]
                print(f'  [{i+1:3d}/{SELFPLAY_EPS_PER_ROUND}] '
                      f'recent_avg={np.mean(recent):.0f} '
                      f'recent_wins={sum(1 for s in recent if s > 900)} '
                      f'ghost={ghost}')

        # Trim buffer
        if len(selfplay_buffer) > SELFPLAY_BUFFER_SIZE:
            selfplay_buffer = selfplay_buffer[-SELFPLAY_BUFFER_SIZE:]

        total_sp_eps += len(round_scores)
        total_wins += round_wins
        dt = time.time() - t0
        print(f'  Round done: avg={np.mean(round_scores):.0f} wins={round_wins} '
              f'in {dt:.0f}s ({len(round_scores)/dt*60:.1f} ep/min)')

        # Save self-play data checkpoint
        sp_path = os.path.join(PROJECT, 'data', f'selfplay_r{round_num:03d}.npz')
        np.savez_compressed(sp_path, trajectories=np.array(selfplay_buffer, dtype=object))

        # ── Retrain ──
        print(f'\n  Retraining on: base({len(base_trajs)}) + selfplay({len(selfplay_buffer)})')
        train_trajs = base_trajs + selfplay_buffer
        ensemble_models = train_ensemble(train_trajs, device, round_num)
        agent = EnsembleAgent(ensemble_models, device)

        # Log
        log.write(f'R{round_num}: sp_eps={len(round_scores)} avg={np.mean(round_scores):.0f} '
                  f'wins={round_wins} models_val_acc='
                  f'{",".join(f"{m:.3f}" for m in [0.0])}\n')
        log.flush()

        # Save round checkpoint
        for i, m in enumerate(ensemble_models):
            torch.save(m.state_dict(),
                       os.path.join(PROJECT, 'checkpoints', f'selfplay_latest_m{i}.pt'))

        round_num += 1

    log.write(f'=== Stopped {datetime.now()} ===\n')
    log.close()
    print(f'\nStopped after {round_num} rounds. Total: {total_sp_eps} sp eps, {total_wins} wins')

if __name__ == '__main__':
    main()
