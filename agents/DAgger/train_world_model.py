"""Train LSTM World Model: (state_seq, action_seq) → (delta_state, reward, done).

Predicts the CHANGE in state (Δs = s_{t+1} - s_t), which focuses learning on
the stochastic ghost movement rather than static background features.
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

STATE_DIM = 448
SEQ_LEN = 4  # frames of history
BATCH_SIZE = 256
HIDDEN = 128

class WorldModel(nn.Module):
    """LSTM encoder → fc decoder → delta_state, reward, done."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(STATE_DIM + 5, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 128)
        self.delta_head = nn.Linear(128, STATE_DIM)
        self.reward_head = nn.Linear(128, 1)
        self.done_head = nn.Linear(128, 1)

    def forward(self, states, actions):
        # states:  (B, T, STATE_DIM)
        # actions: (B, T) int indices
        B, T = states.shape[:2]
        a_onehot = F.one_hot(actions.long(), num_classes=5).float()
        x = torch.cat([states, a_onehot], dim=-1)  # (B, T, STATE_DIM+5)
        o, _ = self.lstm(x)                         # (B, T, HIDDEN)
        last = F.relu(self.fc(o[:, -1]))            # (B, 128)
        delta = self.delta_head(last)               # (B, STATE_DIM)
        reward = self.reward_head(last)             # (B, 1)
        done = self.done_head(last)                 # (B, 1)
        return delta, reward, done

class TransitionDataset(Dataset):
    def __init__(self, trajs):
        self.samples = []
        for t in trajs:
            s = t['states']
            a = t['actions']
            r = t['rewards']
            T = len(a)
            if T < SEQ_LEN + 1:
                continue
            # Pad states to STATE_DIM
            if s.shape[1] < STATE_DIM:
                p = np.zeros((s.shape[0], STATE_DIM), np.float32)
                p[:, :s.shape[1]] = s; s = p
            for i in range(T - SEQ_LEN):
                s_seq = s[i:i+SEQ_LEN]
                a_seq = a[i:i+SEQ_LEN]
                s_next = s[i+SEQ_LEN]
                r_val = r[i+SEQ_LEN-1]  # reward after action at i+SEQ_LEN-1
                s_curr = s[i+SEQ_LEN-1]
                self.samples.append((s_seq, a_seq, s_next, s_curr, r_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_seq, a_seq, s_next, s_curr, r_val = self.samples[idx]
        delta = s_next - s_curr  # predict change
        return (torch.FloatTensor(s_seq),
                torch.LongTensor(a_seq),
                torch.FloatTensor(delta),
                torch.FloatTensor([r_val]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load merged data
data_path = os.path.join(PROJECT, 'data', 'dqn_v4_train.npz')
d = np.load(data_path, allow_pickle=True)
trajs = list(d['trajectories'])
print(f'Loaded {len(trajs)} trajectories')

# Create train/val datasets
np.random.seed(42); np.random.shuffle(trajs)
split = int(len(trajs) * 0.9)
train_ds = TransitionDataset(trajs[:split])
val_ds = TransitionDataset(trajs[split:])
print(f'Train samples: {len(train_ds)}  Val samples: {len(val_ds)}')
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, BATCH_SIZE)

model = WorldModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float('inf')

for epoch in range(100):
    model.train()
    train_losses = []
    for s_seq, a_seq, delta, reward in train_loader:
        s_seq, a_seq = s_seq.to(device), a_seq.to(device)
        delta, reward = delta.to(device), reward.to(device)
        pred_delta, pred_reward, _ = model(s_seq, a_seq)
        loss_delta = F.mse_loss(pred_delta, delta)
        loss_reward = F.mse_loss(pred_reward, reward)
        loss = loss_delta + 0.1 * loss_reward
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for s_seq, a_seq, delta, reward in val_loader:
            s_seq, a_seq = s_seq.to(device), a_seq.to(device)
            delta, reward = delta.to(device), reward.to(device)
            pred_delta, pred_reward, _ = model(s_seq, a_seq)
            loss = F.mse_loss(pred_delta, delta) + 0.1 * F.mse_loss(pred_reward, reward)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints', 'world_model_best.pt'))

    if epoch % 10 == 0:
        print(f'E{epoch:3d} | train={np.mean(train_losses):.4f} val={val_loss:.4f}'
              f' | Δs={F.mse_loss(pred_delta, delta):.4f} r={F.mse_loss(pred_reward, reward):.4f}')

final_path = os.path.join(PROJECT, 'checkpoints', 'world_model_final.pt')
torch.save(model.state_dict(), final_path)
print(f'\nBest val loss: {best_val_loss:.4f}')
print(f'Saved: {final_path}')
