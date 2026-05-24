"""HRA — Paper-accurate implementation.

Key HRA paper design (NIPS 2017):
  1. Each reward component has its own Q-head
  2. Head activates ONLY when its object exists: Q_k=0 if object absent
  3. Q_HRA(s,a) = Σ mask_k(s) × Q_k(s,a)
  4. This forces each head to specialise in its own sub-problem

Heads:
  10 food sector heads — activate when pellets exist in sector
   2 ghost danger heads — always active (ghosts always present)
   2 ghost prey heads — activate when ghost is scared (scaredTimer > 0)
   1 capsule head — activate when capsules remain on board
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout

H, W, C = 11, 20, 8
N_FOOD = 10
N_GHOSTS = 2
N_HEADS = N_FOOD + N_GHOSTS * 2 + 1  # 15

# ── Precompute food sectors ──
def build_food_sectors():
    """5 columns × 2 rows = 10 sectors. Returns sector_mask[H][W] → sector_id or -1."""
    col_bounds = [0, 4, 8, 12, 16, 20]
    row_bounds = [0, 6, 11]
    sector = np.full((H, W), -1, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            ci = 0
            for c in range(len(col_bounds) - 1):
                if col_bounds[c] <= x < col_bounds[c + 1]: ci = c; break
            ri = 0
            for r in range(len(row_bounds) - 1):
                if row_bounds[r] <= y < row_bounds[r + 1]: ri = r; break
            sector[y, x] = ri * 5 + ci
    return sector

FOOD_SECTOR = build_food_sectors()  # (H, W) → sector_id

# ── Walls ──
def get_walls_grid():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg

WG = get_walls_grid()

# ── Flat → Grid ──
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
        g[3 + i, gy, gx] = 1.0
        g[5 + i, gy, gx] = feat[6 + i]
    g[7] = WG; return g

# ── Compute activation masks from grid ──
def grid_masks(grid):
    """Given grid (C, H, W), return mask vector (N_HEADS,) per sample.
    mask[k] = 1.0 if head k's object exists in this state, else 0.0.
    """
    masks = np.zeros(N_HEADS, dtype=np.float32)

    # Food heads: active if any food exists in the sector
    food_grid = grid[0]  # (H, W)
    for y in range(H):
        for x in range(W):
            if food_grid[y, x] > 0.5:
                s = FOOD_SECTOR[y, x]
                if s >= 0:
                    masks[s] = 1.0

    # Ghost danger heads (indices 10, 11): always active
    masks[N_FOOD:N_FOOD + N_GHOSTS] = 1.0

    # Ghost prey heads (indices 12, 13): active if ghost scared
    for gi in range(N_GHOSTS):
        scared = grid[5 + gi].max()  # max scared timer in that ghost's channel
        if scared > 0.01:
            masks[N_FOOD + N_GHOSTS + gi] = 1.0

    # Capsule head (index 14): active if any capsule exists
    if grid[1].max() > 0.5:
        masks[-1] = 1.0

    return masks

# ── HRA Model ──
class HRA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.shared_fc = nn.Linear(64, 128)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 5))
            for _ in range(N_HEADS)
        ])

    def forward(self, x, masks=None):
        """x: (B, C, H, W), masks: (B, N_HEADS) or None.
        Returns q_total (B, 5) and q_heads (B, N_HEADS, 5).
        """
        B = x.size(0)
        f = self.conv(x).mean(dim=[2, 3])
        f = F.relu(self.shared_fc(f))
        q_heads = torch.stack([h(f) for h in self.heads], dim=1)  # (B, N_HEADS, 5)

        if masks is not None:
            masks = masks.to(x.device)
            q_total = (q_heads * masks.unsqueeze(-1)).sum(dim=1)  # (B, 5)
        else:
            q_total = q_heads.sum(dim=1)
        return q_total, q_heads

# ── Dataset (stores masks alongside grids) ──
class HRA_Dataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s, a = t['states'], t['actions']
            T = len(a)
            if T < 2: continue
            for i in range(T - 1):
                masks = grid_masks(s[i])
                self.samples.append((s[i], int(a[i + 1]), masks))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, a, m = self.samples[idx]
        return (torch.FloatTensor(g), torch.LongTensor([a]),
                torch.FloatTensor(m))

# ── Load data ──
def load_data():
    all_trajs = []
    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']
        T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        all_trajs.append({'states': grids, 'actions': a})

    for fname in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fname)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                all_trajs.append({'states': t['states'], 'actions': t['actions']})

    sp_files = sorted(glob.glob(os.path.join(PROJECT, 'data', 'selfplay_r*.npz')))
    # Only use last 3 self-play files (~300 eps) to keep training fast
    for fp in sp_files[-3:]:
        d = np.load(fp, allow_pickle=True)
        for t in d['trajectories']:
            all_trajs.append({'states': t['states'], 'actions': t['actions']})

    print(f'Loaded {len(all_trajs)} trajectories (last {min(3, len(sp_files))} self-play)')
    return all_trajs

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'HRA: {N_HEADS} heads ({N_FOOD} food + {N_GHOSTS*2} ghost + 1 capsule)')
    print(f'Training: CE on masked Q_total, each head only active when object exists')

    trajs = load_data()
    np.random.seed(42); np.random.shuffle(trajs)
    split = int(len(trajs) * 0.9)
    train_ds = HRA_Dataset(trajs[:split])
    val_ds = HRA_Dataset(trajs[split:])
    train_loader = DataLoader(train_ds, 256, shuffle=True)
    val_loader = DataLoader(val_ds, 256)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    for model_id in range(5):
        print(f'\n{"="*50}\nHRA Model {model_id+1}/5\n{"="*50}')
        torch.manual_seed(model_id); np.random.seed(model_id)

        model = HRA_Model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_val = 0.0

        for epoch in range(100):
            model.train()
            t_loss, t_acc = [], []
            for grids, actions, masks in train_loader:
                grids = grids.to(device)
                actions = actions.to(device)
                q_total, q_heads = model(grids, masks)
                loss = F.cross_entropy(q_total, actions.squeeze(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                t_loss.append(loss.item())
                t_acc.append((q_total.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            model.eval()
            v_acc = []
            with torch.no_grad():
                for grids, actions, masks in val_loader:
                    grids, actions = grids.to(device), actions.to(device)
                    q_total, _ = model(grids, masks)
                    v_acc.append((q_total.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            val_acc = np.mean(v_acc)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints', f'hra_m{model_id}_best.pt'))

            if epoch % 20 == 0:
                active_heads = masks.sum(dim=1).mean().item()
                print(f'E{epoch:3d} | loss={np.mean(t_loss):.4f} '
                      f'acc={np.mean(t_acc):.3f} | val_acc={val_acc:.3f} '
                      f'active_heads={active_heads:.1f}')

        torch.save(model.state_dict(),
                   os.path.join(PROJECT, 'checkpoints', f'hra_m{model_id}_final.pt'))
        print(f'Model {model_id}: best_val_acc={best_val:.3f}')

    print('\nAll HRA models trained.')

if __name__ == '__main__':
    main()
