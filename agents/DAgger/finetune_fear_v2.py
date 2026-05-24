"""DAgger R1 fine-tune: only inject DANGER frames from Void data.

Strategy: filter Void trajectories to frames where ghost is within 5 steps.
Only these "fear-critical" frames get high weight. Safe frames are ignored.
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout

H, W, C = 11, 20, 8
BATCH = 256
EPOCHS = 40
LR = 3e-5

def get_walls():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg
WG = get_walls()

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

def min_ghost_dist(grid):
    pac = np.argwhere(grid[2] > 0.5)
    if len(pac) == 0: return 0
    py, px = pac[0]; md = 999
    for gi in range(2):
        gp = np.argwhere(grid[3 + gi] > 0.5)
        if len(gp) > 0:
            d = abs(py - gp[0][0]) + abs(px - gp[0][1])
            if d < md: md = d
    return md

class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

class SimpleDataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s, a = t['states'], t['actions']
            w = float(t.get('weight', 1.0))
            T = len(a)
            if T < 2: continue
            for i in range(T - 1):
                self.samples.append((s[i], int(a[i + 1]), w))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, a, w = self.samples[idx]
        return torch.FloatTensor(g), torch.LongTensor([a]), torch.FloatTensor([w])

def load_data():
    all_trajs = []

    # Original (1x) - all frames
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']; T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        all_trajs.append({'states': grids, 'actions': a, 'weight': 1.0})

    # DAgger data (2x)
    for fn in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fn)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 2.0})

    # Self-play (2x)
    for fp in sorted(glob.glob(os.path.join(PROJECT, 'data', 'selfplay_r*.npz')))[-3:]:
        d = np.load(fp, allow_pickle=True)
        for t in d['trajectories']:
            all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 2.0})

    # Void: ONLY danger frames (ghost < 5 steps)
    vp = os.path.join(PROJECT, 'data', 'void_expert.npz')
    if os.path.exists(vp):
        d = np.load(vp, allow_pickle=True)
        void_danger = 0; void_total = 0
        for t in d['trajectories']:
            s = t['states']; a = t['actions']; T = len(a)
            if T < 2: continue
            danger_grids = []; danger_actions = []
            for i in range(T):
                if min_ghost_dist(s[i]) < 5:
                    danger_grids.append(s[i])
                    if i + 1 < T: danger_actions.append(int(a[i + 1]))
                    void_danger += 1
                void_total += 1
            if len(danger_actions) >= 2:
                all_trajs.append({'states': np.array(danger_grids),
                                  'actions': np.array(danger_actions), 'weight': 5.0})
        print(f'Void danger frames: {void_danger}/{void_total} ({void_danger/void_total*100:.0f}%) — injected at 5x')

    # Attack: ONLY hunt frames (scared ghost nearby)
    ap = os.path.join(PROJECT, 'data', 'attack_expert.npz')
    if os.path.exists(ap):
        d = np.load(ap, allow_pickle=True)
        attack_hunt = 0; attack_total = 0
        for t in d['trajectories']:
            s = t['states']; a = t['actions']; T = len(a)
            if T < 2: continue
            hunt_grids = []; hunt_actions = []
            for i in range(T):
                any_scared = s[i][5 + 0].max() > 0.1 or s[i][5 + 1].max() > 0.1
                if any_scared:
                    hunt_grids.append(s[i])
                    if i + 1 < T: hunt_actions.append(int(a[i + 1]))
                    attack_hunt += 1
                attack_total += 1
            if len(hunt_actions) >= 2:
                all_trajs.append({'states': np.array(hunt_grids),
                                  'actions': np.array(hunt_actions), 'weight': 5.0})
        print(f'Attack hunt frames: {attack_hunt}/{attack_total} ({attack_hunt/attack_total*100:.0f}%) — injected at 5x')

    print(f'Total: {len(all_trajs)} trajectory segments')
    return all_trajs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\nTargeted fine-tune: only danger/hunt frames at high weight\n')

    all_trajs = load_data()
    np.random.seed(42); np.random.shuffle(all_trajs)
    split = int(len(all_trajs) * 0.9)
    train_ds = SimpleDataset(all_trajs[:split])
    val_ds = SimpleDataset(all_trajs[split:])
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}\n')

    base_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    pretrained = torch.load(base_path, map_location='cpu')

    for mid in range(5):
        print(f'Fear v2 Model {mid+1}/5')
        model = CNNDQN().to(device)
        model.load_state_dict(pretrained)

        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        best_val = 0.0

        for epoch in range(EPOCHS):
            model.train()
            t_loss, t_acc = [], []
            for grids, actions, weights in train_loader:
                grids, actions = grids.to(device), actions.to(device)
                weights = weights.to(device)
                q = model(grids)
                loss_per = F.cross_entropy(q, actions.squeeze(-1), reduction='none')
                loss = (loss_per * weights).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                t_loss.append(loss.item())
                t_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            model.eval(); v_acc = []
            with torch.no_grad():
                for grids, actions, _ in val_loader:
                    grids, actions = grids.to(device), actions.to(device)
                    q = model(grids)
                    v_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            val_acc = np.mean(v_acc)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints', f'fear2_m{mid}_best.pt'))
            if epoch % 10 == 0:
                print(f'  E{epoch:3d}: loss={np.mean(t_loss):.4f} acc={np.mean(t_acc):.3f} val={val_acc:.3f}')

        torch.save(model.state_dict(),
                   os.path.join(PROJECT, 'checkpoints', f'fear2_m{mid}_final.pt'))
        print(f'  best_val={best_val:.3f}')

    print('\nDone.')

if __name__ == '__main__':
    main()
