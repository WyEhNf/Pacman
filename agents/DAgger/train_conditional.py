"""CSCNN v2 — Soft-modes + warm-start from best model.

Architecture:
  DAgger R1 CNN (frozen) → 64d features
    ├─ FC_SAFE(128→5)  × safe_weight
    ├─ FC_DANGER(128→5) × danger_weight
    └─ FC_HUNT(128→5)  × hunt_weight
  Q = weighted sum

All FC heads initialized from DAgger fc weights.
Soft mode weights prevent boundary oscillation.
"""
import sys, os, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout

H, W, C = 11, 20, 8
BATCH = 256
EPOCHS = 80
LR_CNN = 1e-5   # tiny LR for backbone
LR_FC = 1e-3     # normal LR for heads

# ── Walls ──
def get_walls():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo

WG, LO = get_walls()

# ── Soft mode weights ──
def soft_mode_weights(grid):
    """Continuous mode weights — no hard boundary."""
    pacman = np.argwhere(grid[2] > 0.5)
    if len(pacman) == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    py, px = pacman[0]

    min_dist = 999.0
    max_scared = 0.0

    for gi in range(2):
        scared = grid[5 + gi].max()
        if scared > max_scared:
            max_scared = scared
        ghost_pos = np.argwhere(grid[3 + gi] > 0.5)
        if len(ghost_pos) > 0:
            d = abs(py - ghost_pos[0][0]) + abs(px - ghost_pos[0][1])
            if d < min_dist: min_dist = d

    # Danger: 1.0 when ghost adjacent, 0.0 when far away
    danger = np.clip(5.0 / max(min_dist, 0.5), 0.0, 1.0)
    # Hunt: proportional to scared timer
    hunt = np.clip(max_scared / 40.0, 0.0, 1.0)
    # If actively hunting, reduce danger (scared ghosts aren't dangerous)
    if hunt > 0.1:
        danger *= (1.0 - hunt)
    # Safe: remainder
    safe = np.clip(1.0 - danger - hunt, 0.0, 1.0)

    return np.array([safe, danger, hunt], dtype=np.float32)

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
        g[3 + i, gy, gx] = 1.0
        g[5 + i, gy, gx] = feat[6 + i]
    g[7] = WG; return g

# ── Model ──
class CSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Three mode-specific FC heads
        self.fc_safe = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
        self.fc_danger = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
        self.fc_hunt = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))

    def forward(self, x, mode_weights):
        f = self.conv(x).mean(dim=[2, 3])
        q_safe = self.fc_safe(f)
        q_danger = self.fc_danger(f)
        q_hunt = self.fc_hunt(f)
        # Soft weighted sum
        w = mode_weights.unsqueeze(-1)  # (B, 3, 1)
        q_all = torch.stack([q_safe, q_danger, q_hunt], dim=1)  # (B, 3, 5)
        return (q_all * w).sum(dim=1)

# ── Dataset ──
class CondDataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s, a = t['states'], t['actions']
            w = float(t.get('weight', 1.0))
            T = len(a)
            if T < 2: continue
            for i in range(T - 1):
                mw = soft_mode_weights(s[i])
                self.samples.append((s[i], int(a[i + 1]), mw, w))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        g, a, mw, w = self.samples[idx]
        return (torch.FloatTensor(g), torch.LongTensor([a]),
                torch.FloatTensor(mw), torch.FloatTensor([w]))

# ── Load data ──
def load_data():
    all_trajs = []

    old = np.load(os.path.join(PROJECT, 'data', 'dqn_v5_train.npz'), allow_pickle=True)
    for t in old['trajectories']:
        sf, a = t['states'], t['actions']; T = len(a)
        if T < 2: continue
        if sf.shape[1] < 448:
            p = np.zeros((sf.shape[0], 448), np.float32); p[:, :sf.shape[1]] = sf; sf = p
        grids = np.stack([flat_to_grid(sf[i]) for i in range(T)])
        all_trajs.append({'states': grids, 'actions': a, 'weight': 1.0})

    for fn in ['dagger_trajectories.npz', 'dagger2_trajectories.npz']:
        fp = os.path.join(PROJECT, 'data', fn)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            for t in d['trajectories']:
                all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 2.0})

    for fp in sorted(glob.glob(os.path.join(PROJECT, 'data', 'selfplay_r*.npz')))[-3:]:
        d = np.load(fp, allow_pickle=True)
        for t in d['trajectories']:
            all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 2.0})

    vp = os.path.join(PROJECT, 'data', 'void_expert.npz')
    if os.path.exists(vp):
        d = np.load(vp, allow_pickle=True)
        for t in d['trajectories']:
            all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 3.0})

    ap = os.path.join(PROJECT, 'data', 'attack_expert.npz')
    if os.path.exists(ap):
        d = np.load(ap, allow_pickle=True)
        for t in d['trajectories']:
            all_trajs.append({'states': t['states'], 'actions': t['actions'], 'weight': 3.0})

    print(f'Loaded {len(all_trajs)} trajectories')
    return all_trajs

# ── Main ──
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    trajs = load_data()
    np.random.seed(42); np.random.shuffle(trajs)
    split = int(len(trajs) * 0.9)
    train_ds = CondDataset(trajs[:split])
    val_ds = CondDataset(trajs[split:])
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    # Load pretrained CNN
    base_path = os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt')
    pretrained = torch.load(base_path, map_location='cpu')

    for model_id in range(5):
        print(f'\n{"="*50}\nCSCNN v2 Model {model_id+1}/5\n{"="*50}')
        torch.manual_seed(model_id); np.random.seed(model_id)

        model = CSCNN().to(device)

        # Warm-start CNN from pretrained
        conv_state = {k.replace('conv.', ''): v for k, v in pretrained.items() if k.startswith('conv.')}
        model.conv.load_state_dict(conv_state)

        # Initialize FC heads from pretrained fc
        fc_state = {k.replace('fc.', ''): v for k, v in pretrained.items() if k.startswith('fc.')}
        model.fc_safe.load_state_dict(fc_state)
        model.fc_danger.load_state_dict(fc_state)
        model.fc_hunt.load_state_dict(fc_state)
        print(f'Warm-started from {base_path}')

        # Separate LR for backbone and heads
        opt = torch.optim.Adam([
            {'params': model.conv.parameters(), 'lr': LR_CNN},
            {'params': model.fc_safe.parameters(), 'lr': LR_FC},
            {'params': model.fc_danger.parameters(), 'lr': LR_FC},
            {'params': model.fc_hunt.parameters(), 'lr': LR_FC},
        ], weight_decay=1e-5)

        best_val = 0.0

        for epoch in range(EPOCHS):
            model.train()
            t_loss, t_acc = [], []
            for grids, actions, modes, weights in train_loader:
                grids = grids.to(device)
                actions = actions.to(device)
                modes = modes.to(device)
                weights = weights.to(device)

                q = model(grids, modes)
                loss_per = F.cross_entropy(q, actions.squeeze(-1), reduction='none')
                loss = (loss_per * weights).mean()

                opt.zero_grad(); loss.backward(); opt.step()
                t_loss.append(loss.item())
                t_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            model.eval()
            v_acc = []
            with torch.no_grad():
                for grids, actions, modes, _ in val_loader:
                    grids = grids.to(device)
                    actions = actions.to(device)
                    modes = modes.to(device)
                    q = model(grids, modes)
                    v_acc.append((q.argmax(-1) == actions.squeeze(-1)).float().mean().item())

            val_acc = np.mean(v_acc)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints', f'cs_m{model_id}_best.pt'))

            if epoch % 20 == 0:
                # Show mode specialization
                safe_act = model.fc_safe[0].weight.abs().mean().item()
                danger_act = model.fc_danger[0].weight.abs().mean().item()
                hunt_act = model.fc_hunt[0].weight.abs().mean().item()
                print(f'E{epoch:3d} | loss={np.mean(t_loss):.4f} '
                      f'acc={np.mean(t_acc):.3f} | val={val_acc:.3f} '
                      f'|W| S={safe_act:.2f} D={danger_act:.2f} H={hunt_act:.2f}')

        torch.save(model.state_dict(),
                   os.path.join(PROJECT, 'checkpoints', f'cs_m{model_id}_final.pt'))
        print(f'Model {model_id}: best_val_acc={best_val:.3f}')

    print('\nAll CSCNN v2 models trained.')

if __name__ == '__main__':
    main()
