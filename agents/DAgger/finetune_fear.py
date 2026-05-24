"""Fine-tune DAgger R1 with Void + Attack data to inject ghost awareness.

Low LR, weighted data. Preserves existing eating skill while learning fear + hunting.
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout

H, W, C = 11, 20, 8
BATCH = 256
EPOCHS = 50
LR_CNN = 1e-5
LR_FC = 5e-5

# ── Walls ──
def get_walls():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg

WG = get_walls()

# ── Flat → Grid ──
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

# ── Dataset ──
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

# ── Load data ──
def load_data():
    trajs = []
    # Original (1x)
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']; T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        trajs.append({'states': grids, 'actions': a, 'weight': 1.0})

    # Void (3x) — survival under aggressive ghosts
    vp = os.path.join(PROJECT, 'data', 'void_expert.npz')
    if os.path.exists(vp):
        d = np.load(vp, allow_pickle=True)
        for t in d['trajectories']:
            trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 3.0})

    # Attack (3x) — ghost hunting
    ap = os.path.join(PROJECT, 'data', 'attack_expert.npz')
    if os.path.exists(ap):
        d = np.load(ap, allow_pickle=True)
        for t in d['trajectories']:
            trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 3.0})

    print(f'Data: {len(trajs)} trajectories (original + Void×3 + Attack×3)')
    return trajs

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\nFine-tune DAgger R1 with Void+Attack, LR_CNN={LR_CNN}, LR_FC={LR_FC}\n')

    trajs = load_data()
    np.random.seed(42); np.random.shuffle(trajs)
    split = int(len(trajs) * 0.9)
    train_ds = SimpleDataset(trajs[:split])
    val_ds = SimpleDataset(trajs[split:])
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    base_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    pretrained = torch.load(base_path, map_location='cpu')

    for mid in range(5):
        print(f'\n{"="*50}\nFear Model {mid+1}/5\n{"="*50}')

        model = CNNDQN().to(device)
        model.load_state_dict(pretrained)
        print(f'Loaded from DAgger R1')

        opt = torch.optim.Adam([
            {'params': model.conv.parameters(), 'lr': LR_CNN},
            {'params': model.fc.parameters(), 'lr': LR_FC},
        ], weight_decay=1e-5)

        best_val = 0.0
        for epoch in range(EPOCHS):
            model.train()
            t_loss, t_acc = [], []
            for grids, actions, weights in train_loader:
                grids = grids.to(device)
                actions = actions.to(device)
                weights = weights.to(device)
                q = model(grids)
                loss_per = F.cross_entropy(q, actions.squeeze(-1), reduction='none')
                loss = (loss_per * weights).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                t_loss.append(loss.item())
                t_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            model.eval()
            v_acc = []
            with torch.no_grad():
                for grids, actions, _ in val_loader:
                    grids, actions = grids.to(device), actions.to(device)
                    q = model(grids)
                    v_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            val_acc = np.mean(v_acc)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints', f'fear_m{mid}_best.pt'))

            if epoch % 10 == 0:
                print(f'E{epoch:3d} | loss={np.mean(t_loss):.4f} '
                      f'acc={np.mean(t_acc):.3f} | val={val_acc:.3f}')

        torch.save(model.state_dict(),
                   os.path.join(PROJECT, 'checkpoints', f'fear_m{mid}_final.pt'))
        print(f'Model {mid}: best_val={best_val:.3f}')

    print('\nAll Fear models fine-tuned.')

if __name__ == '__main__':
    main()
