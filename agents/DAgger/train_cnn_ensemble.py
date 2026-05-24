"""Train CNN DQN Ensemble — 5 models, different seeds, CE-only supervised learning.

State representation: 8-channel 2D grid (H×W) instead of flat 448-dim vector.
  Ch0: Food    Ch1: Capsules   Ch2: Pacman
  Ch3-4: Ghost positions   Ch5-6: Ghost scared timers   Ch7: Walls

Ensemble averages Q-values from all models during inference.
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# ── Config ──
H, W = 11, 20  # mediumClassic grid
N_CHANNELS = 8
N_MODELS = 5
BATCH = 256
EPOCHS = 150

# ── CNN DQN ──
class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        o = self.conv(x)           # (B, 64, H, W)
        o = o.mean(dim=[2, 3])     # global average pool → (B, 64)
        return self.fc(o)          # (B, 5)

# ── Convert flat features → 2D grid ──
def flat_to_grid(features, walls_grid):
    """Convert 448-dim flat features to (C, H, W) grid.
    Feature layout: [pacman(2), ghost_pos(2*2), ghost_scared(2), food(220), capsules(220)]
    """
    grid = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    feat = features

    # Food (channel 0)
    grid[0] = feat[8:8+H*W].reshape(W, H).T

    # Capsules (channel 1)
    grid[1] = feat[228:228+H*W].reshape(W, H).T

    # Pacman position (channel 2) — px/H, py/W (swapped normalization)
    px = min(max(int(feat[0] * H), 0), W - 1)
    py = min(max(int(feat[1] * W), 0), H - 1)
    grid[2, py, px] = 1.0

    # Ghosts (channels 3-6)
    for i in range(2):
        gx = min(max(int(feat[2 + i*2] * H), 0), W - 1)
        gy = min(max(int(feat[3 + i*2] * W), 0), H - 1)
        grid[3 + i, gy, gx] = 1.0
        grid[5 + i, gy, gx] = feat[6 + i]

    # Walls (channel 7)
    if walls_grid is not None:
        grid[7] = walls_grid

    return grid

# ── Dataset ──
class GridDataset(Dataset):
    def __init__(self, trajs, walls_grid):
        self.samples = []
        for t in trajs:
            s = t['states']
            a = t['actions']
            T = len(a)
            if T < 2: continue
            if s.shape[1] < 448:
                p = np.zeros((s.shape[0], 448), np.float32)
                p[:, :s.shape[1]] = s; s = p
            for i in range(T - 1):
                grid = flat_to_grid(s[i], walls_grid)
                self.samples.append((grid, int(a[i+1])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        g, a = self.samples[idx]
        return torch.FloatTensor(g), torch.LongTensor([a])

# ── Load walls grid ──
def get_walls_grid():
    SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
    sys.path.insert(0, SKEL)
    os.chdir(SKEL)
    import layout
    lo = layout.getLayout('mediumClassic')
    W_w, H_w = lo.walls.width, lo.walls.height
    w_grid = np.zeros((H_w, W_w), dtype=np.float32)
    for x in range(W_w):
        for y in range(H_w):
            if lo.walls.data[x][y]:
                w_grid[y, x] = 1.0
    return w_grid

# ── Main ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load data
d = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
print(f'Training on dqn_v5_train.npz (470 eps)')
trajs = list(d['trajectories'])
np.random.seed(42); np.random.shuffle(trajs)
split = int(len(trajs) * 0.9)

walls_grid = get_walls_grid()
print(f'Walls grid: {walls_grid.shape}')

train_ds = GridDataset(trajs[:split], walls_grid)
val_ds = GridDataset(trajs[split:], walls_grid)
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader = DataLoader(val_ds, BATCH)

# Train N_MODELS with different seeds
for model_id in range(N_MODELS):
    print(f'\n{"="*50}\nTraining CNN Model {model_id+1}/{N_MODELS}\n{"="*50}')
    torch.manual_seed(model_id)
    np.random.seed(model_id)

    model = CNNDQN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_losses, train_accs = [], []
        for grids, actions in train_loader:
            grids, actions = grids.to(device), actions.to(device)
            q = model(grids)
            loss = F.cross_entropy(q, actions.squeeze(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            train_accs.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

        model.eval()
        val_accs = []
        with torch.no_grad():
            for grids, actions in val_loader:
                grids, actions = grids.to(device), actions.to(device)
                q = model(grids)
                val_accs.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

        val_acc = np.mean(val_accs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(PROJECT, 'checkpoints', f'v5_cnn_m{model_id}_best.pt')
            torch.save(model.state_dict(), ckpt)

        if epoch % 20 == 0:
            print(f'E{epoch:3d} | loss={np.mean(train_losses):.4f} '
                  f'acc={np.mean(train_accs):.3f} | val_acc={val_acc:.3f}')

    final = os.path.join(PROJECT, 'checkpoints', f'v5_cnn_m{model_id}_final.pt')
    torch.save(model.state_dict(), final)
    print(f'Model {model_id}: best_val_acc={best_val_acc:.3f}  saved to {final}')

print('\nAll CNN models trained.')
