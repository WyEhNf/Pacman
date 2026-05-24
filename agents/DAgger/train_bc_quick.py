"""Quick BC training — run from command line."""
import sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
import torch, numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import TrajectoryDataset
from src.model.decision_transformer import DecisionTransformer

data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/inc_0100.npz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

ds = TrajectoryDataset(data_path, context_len=20)
loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

model = DecisionTransformer(
    state_dim=ds.state_dim, act_dim=5,
    d_model=256, n_heads=4, n_layers=4, context_len=20,
    dropout=0.1
).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(30):
    losses = []
    for rtg, states, actions, mask in loader:
        rtg, states, actions, mask = rtg.to(device), states.to(device), actions.to(device), mask.to(device)
        B = rtg.shape[0]
        timesteps = torch.zeros(B, 20, dtype=torch.long, device=device)
        logits, _, _ = model(rtg, states, actions, timesteps)
        loss = criterion(logits[mask.bool()].view(-1, 5), actions[mask.bool()].view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f'Epoch {epoch+1:2d}/30  loss={np.mean(losses):.4f}')

ckpt = 'checkpoints/dt_v1_100ep.pt'
os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'state_dim': ds.state_dim,
    'rtg_min': ds.rtg_min,
    'rtg_max': ds.rtg_max,
}, ckpt)
print(f'Saved: {ckpt}')
