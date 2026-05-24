"""Dyna-DQN: CE loss on real data + model-based TD regularization.

Uses LSTM world model to provide temporal consistency signal:
  Q(s,a) ≈ r_wm + γ * max Q(s', a')

TD loss is kept small (λ=0.01) to prevent Q-value explosion.
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

STATE_DIM = 448
SEQ_LEN = 4
BATCH = 128
GAMMA = 0.95
TD_WEIGHT = 0.02  # small regularization weight

# ── World Model (frozen during DQN training) ──

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(STATE_DIM + 5, 128, batch_first=True)
        self.fc = nn.Linear(128, 128)
        self.delta_head = nn.Linear(128, STATE_DIM)
        self.reward_head = nn.Linear(128, 1)
        self.done_head = nn.Linear(128, 1)

    def forward(self, states, actions):
        B, T = states.shape[:2]
        a_onehot = F.one_hot(actions.long(), num_classes=5).float()
        x = torch.cat([states, a_onehot], dim=-1)
        o, _ = self.lstm(x)
        last = F.relu(self.fc(o[:, -1]))
        return self.delta_head(last), self.reward_head(last), self.done_head(last)

# ── DQN ──

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 5)
    def forward(self, x):
        return self.q_head(F.relu(self.fc2(F.relu(self.fc1(x)))))

# ── Dataset ──

class TrajDataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s = t['states']; a = t['actions']; r = t['rewards']
            T = len(a)
            if T < SEQ_LEN + 1: continue
            if s.shape[1] < STATE_DIM:
                p = np.zeros((s.shape[0], STATE_DIM), np.float32)
                p[:, :s.shape[1]] = s; s = p
            for i in range(T - SEQ_LEN):
                s_seq = s[i:i+SEQ_LEN]
                a_seq = a[i:i+SEQ_LEN]
                a_next = a[i+SEQ_LEN]       # expert's next action
                r_val = r[i+SEQ_LEN-1]
                self.samples.append((s_seq, a_seq, a_next, r_val))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s_seq, a_seq, a_next, r_val = self.samples[idx]
        return (torch.FloatTensor(s_seq), torch.LongTensor(a_seq),
                torch.LongTensor([a_next]), torch.FloatTensor([r_val]))

# ── Main ──

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load data
d = np.load(os.path.join(PROJECT, 'data', 'dqn_v4_train.npz'), allow_pickle=True)
trajs = list(d['trajectories'])
print(f'Loaded {len(trajs)} trajectories')

np.random.seed(42); np.random.shuffle(trajs)
split = int(len(trajs) * 0.9)
train_ds = TrajDataset(trajs[:split])
val_ds = TrajDataset(trajs[split:])
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader = DataLoader(val_ds, BATCH)

# Load world model
wm = WorldModel().to(device)
wm.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints', 'world_model_best.pt'), map_location=device))
wm.eval()
for p in wm.parameters():
    p.requires_grad = False
print('World model loaded (frozen)')

# DQN
dqn = DQN().to(device)
target_dqn = DQN().to(device)  # target network for stable TD
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()
opt = torch.optim.Adam(dqn.parameters(), lr=1e-3, weight_decay=1e-5)
best_val_acc = 0.0

for epoch in range(200):
    # Train
    dqn.train()
    ce_losses, td_losses = [], []
    train_accs = []
    for s_seq, a_seq, a_next, r_val in train_loader:
        s_seq, a_seq = s_seq.to(device), a_seq.to(device)
        a_next = a_next.to(device)
        r_val = r_val.to(device)
        B = s_seq.size(0)

        # ── CE loss (supervised on expert actions) ──
        s_curr = s_seq[:, -1]  # last state in sequence
        q_all = dqn(s_curr)    # (B, 5)
        ce_loss = F.cross_entropy(q_all, a_next.squeeze(-1))

        # ── TD regularization via world model ──
        with torch.no_grad():
            delta_s, r_pred, done_pred = wm(s_seq, a_seq)
            s_next_wm = s_curr + delta_s                  # (B, STATE_DIM)
            q_next = target_dqn(s_next_wm)                # (B, 5)
            max_q_next = q_next.max(dim=-1).values        # (B,)
            done_flag = torch.sigmoid(done_pred).squeeze()  # (B,)
            td_target = r_pred.squeeze() + GAMMA * max_q_next * (1 - done_flag)
            td_target = torch.clamp(td_target, -200, 500)  # prevent explosion

        # Q(s, a_expert) for the expert's action
        q_expert = q_all.gather(1, a_next).squeeze(-1)       # (B,)
        td_loss = F.mse_loss(q_expert, td_target)

        loss = ce_loss + TD_WEIGHT * td_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10.0)
        opt.step()

        ce_losses.append(ce_loss.item())
        td_losses.append(td_loss.item())
        train_accs.append((q_all.argmax(-1) == a_next.squeeze(-1)).float().mean().item())

    # Validate
    dqn.eval()
    val_accs = []
    for s_seq, a_seq, a_next, r_val in val_loader:
        s_seq = s_seq.to(device)
        a_next = a_next.to(device)
        q_all = dqn(s_seq[:, -1])
        val_accs.append((q_all.argmax(-1) == a_next.squeeze(-1)).float().mean().item())

    # Update target network
    if epoch % 5 == 0:
        target_dqn.load_state_dict(dqn.state_dict())

    val_acc = np.mean(val_accs)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(dqn.state_dict(), os.path.join(PROJECT, 'checkpoints', 'dqn_dyna_best.pt'))

    if epoch % 10 == 0:
        print(f'E{epoch:3d} | CE={np.mean(ce_losses):.4f} TD={np.mean(td_losses):.4f} '
              f'acc={np.mean(train_accs):.3f} | val_acc={val_acc:.3f}')

final_path = os.path.join(PROJECT, 'checkpoints', 'dqn_dyna_final.pt')
torch.save(dqn.state_dict(), final_path)
print(f'\nBest val acc: {best_val_acc:.3f}')
print(f'Saved: {final_path}')
