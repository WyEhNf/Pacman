"""Pre-train DQN on winning expert trajectories, then save checkpoint."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

STATE_DIM = 448
class DualHeadDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(448, 256); self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 5); self.v_head = nn.Linear(128, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x)); h = F.relu(self.fc2(h))
        return self.q_head(h), self.v_head(h)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualHeadDQN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

d = np.load(os.path.join(PROJECT, 'data/wins_only.npz'), allow_pickle=True)
trajs = d['trajectories']
print(f'Loaded {len(trajs)} winning trajectories | device={device}')

model.train()
for epoch in range(100):
    losses, accs = [], []
    for t in trajs:
        s, a, r = t['states'], t['actions'], t['rewards']
        T = len(a)
        if T < 2: continue
        # Pad state to 448 if needed
        if s.shape[1] < 448:
            p = np.zeros((s.shape[0], 448), np.float32)
            p[:, :s.shape[1]] = s; s = p
        s_t = torch.FloatTensor(s).to(device)
        q, _ = model(s_t)
        a_t = torch.LongTensor(a).to(device)

        # Supervised CE loss
        ce_loss = F.cross_entropy(q[:-1], a_t[1:])

        # TD loss on actual transitions
        rewards = torch.clamp(torch.FloatTensor(r).to(device), -100, 100) / 100.0
        q_target = rewards[1:] + 0.99 * q[:-1].max(-1).values.detach()
        q_pred = q[:-1][range(T-1), a_t[1:]]
        td_loss = F.mse_loss(q_pred, q_target)

        loss = ce_loss + 0.5 * td_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        accs.append((q[:-1].argmax(-1) == a_t[1:]).float().mean().item())

    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d}  loss={np.mean(losses):.4f}  acc={np.mean(accs):.3f}')

ckpt = os.path.join(PROJECT, 'checkpoints/dqn_warmstart.pt')
torch.save(model.state_dict(), ckpt)
print(f'Saved: {ckpt}')
