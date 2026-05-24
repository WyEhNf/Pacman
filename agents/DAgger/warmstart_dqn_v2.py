"""Train DQN from high-quality merged data (160 eps, 50% win rate)."""
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
model = DQN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

d = np.load(os.path.join(PROJECT, 'data/overnight_merged.npz'), allow_pickle=True)
trajs = d['trajectories']
print(f'Loaded {len(trajs)} trajectories (50% win rate, avg score ~955)')

for epoch in range(200):
    losses, accs = [], []
    for t in trajs:
        s, a, r = t['states'], t['actions'], t['rewards']
        T = len(a)
        if T < 2: continue
        if s.shape[1] < STATE_DIM:
            p = np.zeros((s.shape[0], STATE_DIM), np.float32)
            p[:, :s.shape[1]] = s; s = p
        s_t = torch.FloatTensor(s).to(device)
        a_t = torch.LongTensor(a).to(device)
        q = model(s_t)
        # CE-only: supervised learning on expert actions (no TD bootstrapping)
        loss = F.cross_entropy(q[:-1], a_t[1:])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        accs.append((q[:-1].argmax(-1) == a_t[1:]).float().mean().item())
    if epoch % 20 == 0:
        print(f'E{epoch:3d} loss={np.mean(losses):.4f} acc={np.mean(accs):.3f}')

ckpt = os.path.join(PROJECT, 'checkpoints/dqn_warmstart_v2.pt')
torch.save(model.state_dict(), ckpt)
print(f'Saved: {ckpt}')
