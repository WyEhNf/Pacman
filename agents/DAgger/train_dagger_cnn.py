"""Train CNN Ensemble on DAgger-augmented data (520 eps, DAgger 3x weighted)."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

H, W, C = 11, 20, 8
N_MODELS = 5
BATCH = 128
EPOCHS = 150

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

class DaggerDataset(Dataset):
    """Grid-native dataset with per-sample DAgger weights."""
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s = t['states']  # (T, C, H, W)
            a = t['actions'] # (T,)
            w = float(t.get('weight', 1.0))
            T = len(a)
            if T < 2: continue
            for i in range(T - 1):
                self.samples.append((s[i], int(a[i + 1]), w))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        g, a, w = self.samples[idx]
        return torch.FloatTensor(g), torch.LongTensor([a]), torch.FloatTensor([w])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

d = np.load(os.path.join(PROJECT, 'data', 'dagger2_merged.npz'), allow_pickle=True)
trajs = list(d['trajectories'])
print(f'Data: {len(trajs)} trajectories')

np.random.seed(42); np.random.shuffle(trajs)
split = int(len(trajs) * 0.9)
train_ds = DaggerDataset(trajs[:split])
val_ds = DaggerDataset(trajs[split:])
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader = DataLoader(val_ds, BATCH)

for model_id in range(N_MODELS):
    print(f'\n{"="*50}\nDAgger CNN Model {model_id+1}/{N_MODELS}\n{"="*50}')
    torch.manual_seed(model_id); np.random.seed(model_id)

    model = CNNDQN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_acc = [], []
        for grids, actions, weights in train_loader:
            grids = grids.to(device)
            actions = actions.to(device)
            weights = weights.to(device)
            q = model(grids)
            # Weighted CE: DAgger samples count 3x
            loss_per_sample = F.cross_entropy(q, actions.squeeze(-1), reduction='none')
            loss = (loss_per_sample * weights).mean()
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(PROJECT, 'checkpoints', f'dagger2_cnn_m{model_id}_best.pt'))

        if epoch % 20 == 0:
            print(f'E{epoch:3d} | loss={np.mean(t_loss):.4f} acc={np.mean(t_acc):.3f} | val_acc={val_acc:.3f}')

    final = os.path.join(PROJECT, 'checkpoints', f'dagger2_cnn_m{model_id}_final.pt')
    torch.save(model.state_dict(), final)
    print(f'Model {model_id}: best_val_acc={best_val_acc:.3f}')

print('\nAll DAgger CNN models trained.')
