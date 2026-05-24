"""CNN → GRU → Dueling Q-Network. CE-only supervised learning.

Architecture:
  CNN(8ch→64) → GAP → 64d per frame  (shared)
  GRU(64→128, 1 layer) → 128d hidden
  Dueling: V(s) + A(s,a) - mean(A)
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# ── Config ──
H, W, C = 11, 20, 8
SEQ_LEN = 3
N_MODELS = 5
BATCH = 128
EPOCHS = 150

# ── Model ──
class GRUDuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial encoder (shared across frames)
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Temporal encoder
        self.gru = nn.GRU(64, 128, batch_first=True)
        # Dueling heads
        self.v_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, H, W) for single frame
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)
        B, T = x.shape[:2]
        # CNN per-frame
        x_flat = x.view(B * T, C, H, W)
        feats = self.conv(x_flat).mean(dim=[2, 3])  # (B*T, 64)
        feats = feats.view(B, T, 64)                 # (B, T, 64)
        # GRU
        _, h = self.gru(feats)                       # h: (1, B, 128)
        h = h.squeeze(0)                             # (B, 128)
        # Dueling
        v = self.v_head(h)                           # (B, 1)
        a = self.a_head(h)                           # (B, 5)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

# ── Feature conversion ──
def flat_to_grid(features, walls_grid):
    grid = np.zeros((C, H, W), dtype=np.float32)
    feat = features
    grid[0] = feat[8:8+H*W].reshape(W, H).T
    grid[1] = feat[228:228+H*W].reshape(W, H).T
    px = min(max(int(feat[0] * H), 0), W - 1)
    py = min(max(int(feat[1] * W), 0), H - 1)
    grid[2, py, px] = 1.0
    for i in range(2):
        gx = min(max(int(feat[2 + i*2] * H), 0), W - 1)
        gy = min(max(int(feat[3 + i*2] * W), 0), H - 1)
        grid[3 + i, gy, gx] = 1.0
        grid[5 + i, gy, gx] = feat[6 + i]
    if walls_grid is not None:
        grid[7] = walls_grid
    return grid

# ── Dataset ──
class SeqGridDataset(Dataset):
    def __init__(self, trajs, walls_grid):
        # Pre-compute all grids in memory (fast GPU training)
        self.samples = []
        for t in trajs:
            s = t['states']; a = t['actions']
            T = len(a)
            if T < SEQ_LEN + 1: continue
            if s.shape[1] < 448:
                p = np.zeros((s.shape[0], 448), np.float32)
                p[:, :s.shape[1]] = s; s = p
            # Convert all states to grids once
            grids = np.zeros((T, C, H, W), dtype=np.float32)
            for i in range(T):
                grids[i] = flat_to_grid(s[i], walls_grid)
            for i in range(T - SEQ_LEN):
                self.samples.append((grids[i:i+SEQ_LEN].copy(), int(a[i+SEQ_LEN])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        grids, action = self.samples[idx]
        return torch.FloatTensor(grids), torch.LongTensor([action])

# ── Walls ──
def get_walls_grid():
    SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
    sys.path.insert(0, SKEL); os.chdir(SKEL)
    import layout
    lo = layout.getLayout('mediumClassic')
    W_w, H_w = lo.walls.width, lo.walls.height
    wg = np.zeros((H_w, W_w), dtype=np.float32)
    for x in range(W_w):
        for y in range(H_w):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg

# ── Main ──
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    d = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    trajs = list(d['trajectories'])
    print(f'Data: {len(trajs)} trajectories')

    np.random.seed(42); np.random.shuffle(trajs)
    split = int(len(trajs) * 0.9)
    walls_grid = get_walls_grid()
    print(f'Walls: {walls_grid.shape}')

    train_ds = SeqGridDataset(trajs[:split], walls_grid)
    val_ds = SeqGridDataset(trajs[split:], walls_grid)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)

    for model_id in range(N_MODELS):
        print(f'\n{"="*50}\nGRU+Dueling Model {model_id+1}/{N_MODELS}\n{"="*50}')
        torch.manual_seed(model_id); np.random.seed(model_id)

        model = GRUDuelingDQN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            t_loss, t_acc = [], []
            for grids, actions in train_loader:
                grids, actions = grids.to(device), actions.to(device)
                q = model(grids)
                loss = F.cross_entropy(q, actions.squeeze(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                t_loss.append(loss.item())
                t_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            model.eval()
            v_acc = []
            with torch.no_grad():
                for grids, actions in val_loader:
                    grids, actions = grids.to(device), actions.to(device)
                    q = model(grids)
                    v_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            val_acc = np.mean(v_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints', f'gruduel_m{model_id}_best.pt'))

            if epoch % 20 == 0:
                print(f'E{epoch:3d} | loss={np.mean(t_loss):.4f} acc={np.mean(t_acc):.3f} | val_acc={val_acc:.3f}')

        final = os.path.join(PROJECT, 'checkpoints', f'gruduel_m{model_id}_final.pt')
        torch.save(model.state_dict(), final)
        print(f'Model {model_id}: best_val_acc={best_val_acc:.3f}')

    print('\nAll GRU+Dueling models trained.')
