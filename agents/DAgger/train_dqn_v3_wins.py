"""Train DQN v3 — CE-only on WINS-ONLY expert data (170 wins, avg=1598)."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

STATE_DIM = 448

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 5)
    def forward(self, x):
        return self.q_head(F.relu(self.fc2(F.relu(self.fc1(x)))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

data_path = os.path.join(PROJECT, 'data', 'dqn_v3_wins.npz')
d = np.load(data_path, allow_pickle=True)
trajs = list(d['trajectories'])

np.random.seed(42)
np.random.shuffle(trajs)
split = int(len(trajs) * 0.9)
train_trajs = trajs[:split]
val_trajs = trajs[split:]

scores = [t['score'] for t in trajs]
print(f'Data: {len(trajs)} wins-only eps (train={len(train_trajs)} val={len(val_trajs)})')
print(f'  avg score={np.mean(scores):.0f}')

model = DQN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
best_val_acc = 0.0

for epoch in range(200):
    model.train()
    train_losses, train_accs = [], []
    for t in train_trajs:
        s, a = t['states'], t['actions']
        T = len(a)
        if T < 2: continue
        if s.shape[1] < STATE_DIM:
            p = np.zeros((s.shape[0], STATE_DIM), np.float32)
            p[:, :s.shape[1]] = s; s = p
        s_t = torch.FloatTensor(s).to(device)
        a_t = torch.LongTensor(a).to(device)
        q = model(s_t)
        loss = F.cross_entropy(q[:-1], a_t[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(loss.item())
        train_accs.append((q[:-1].argmax(-1) == a_t[1:]).float().mean().item())

    model.eval()
    val_losses, val_accs = [], []
    with torch.no_grad():
        for t in val_trajs:
            s, a = t['states'], t['actions']
            T = len(a)
            if T < 2: continue
            if s.shape[1] < STATE_DIM:
                p = np.zeros((s.shape[0], STATE_DIM), np.float32)
                p[:, :s.shape[1]] = s; s = p
            s_t = torch.FloatTensor(s).to(device)
            a_t = torch.LongTensor(a).to(device)
            q = model(s_t)
            loss = F.cross_entropy(q[:-1], a_t[1:])
            val_losses.append(loss.item())
            val_accs.append((q[:-1].argmax(-1) == a_t[1:]).float().mean().item())

    val_acc = np.mean(val_accs)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints', 'dqn_v3_wins_best.pt'))

    if epoch % 10 == 0:
        print(f'E{epoch:3d} | train loss={np.mean(train_losses):.4f} acc={np.mean(train_accs):.3f} '
              f'| val loss={np.mean(val_losses):.4f} acc={val_acc:.3f}')

final_path = os.path.join(PROJECT, 'checkpoints', 'dqn_v3_wins_final.pt')
torch.save(model.state_dict(), final_path)
print(f'\nBest val acc: {best_val_acc:.3f}')
print(f'Saved: {final_path}')
