"""
Smoke test: run the full pipeline end-to-end with minimal settings.

1. Collect 10 expert episodes (fast, shallow agents)
2. Train BC for 3 epochs
3. Evaluate on a single game

Usage:
    cd e:\Pacman
    conda activate pacman
    python scripts/smoke_test.py
"""

import sys, os, tempfile, warnings
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON_M = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT)
sys.path.insert(0, SKELETON_M)
os.chdir(SKELETON_M)

import layout
from pacman import GameState
from game import Directions
import ghostAgents

from src.model.transformer_block import TransformerBlock
from src.model.decision_transformer import DecisionTransformer
from src.data.dataset import TrajectoryDataset
from scripts.collect_expert_data import (
    collect_one_episode, extract_features, get_state_shape, _make_experts
)

warnings.filterwarnings('ignore')

ACTION_MAP_STR = {
    0: Directions.NORTH, 1: Directions.SOUTH,
    2: Directions.EAST, 3: Directions.WEST, 4: Directions.STOP,
}

# ═══════════════════════════════════════════════════════════════════
#  Step 1: Collect a few expert episodes
# ═══════════════════════════════════════════════════════════════════

print("Step 1: Collecting expert episodes (depth=2, fast)...")
layout_obj = layout.getLayout('smallClassic')
state_dim = get_state_shape(layout_obj)
# Build a simple expert agent manually
from multiAgents import AlphaBetaAgent
agent = AlphaBetaAgent(depth='2')

def ghost_factory(layout):
    return [ghostAgents.DirectionalGhost(i + 1, 0.8, 0.8)
            for i in range(layout.getNumGhosts())]

trajectories = []
for ep in range(10):
    traj = collect_one_episode(agent, ghost_factory, layout_obj, min_steps=5)
    if traj is not None:
        trajectories.append(traj)
    print(f"  Episode {ep+1}/10: steps={traj['steps'] if traj else 'N/A'}")

print(f"  Collected {len(trajectories)} episodes, state_dim={state_dim}")

# Save to temp file
tmp_path = os.path.join(tempfile.gettempdir(), 'smoke_trajectories.npz')
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
np.savez_compressed(tmp_path,
                    trajectories=np.array(trajectories, dtype=object))
print(f"  Saved to {tmp_path}")

# ═══════════════════════════════════════════════════════════════════
#  Step 2: Train BC (tiny model, 3 epochs)
# ═══════════════════════════════════════════════════════════════════

print("\nStep 2: BC training (3 epochs, tiny model)...")

ds = TrajectoryDataset(tmp_path, context_len=10, state_dim=state_dim)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

model = DecisionTransformer(
    state_dim=state_dim, act_dim=5,
    d_model=64, n_heads=2, n_layers=3,
    context_len=10, dropout=0.0,
)
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    losses = []
    for rtg, states, actions, mask in loader:
        B = rtg.shape[0]
        timesteps = torch.zeros(B, 10, dtype=torch.long)

        action_logits, _, _ = model(rtg, states, actions, timesteps)
        loss = criterion(
            action_logits[mask.bool()].view(-1, 5),
            actions[mask.bool()].view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")

# ═══════════════════════════════════════════════════════════════════
#  Step 3: Evaluate on one game
# ═══════════════════════════════════════════════════════════════════

print("\nStep 3: Evaluating on one game...")
model.eval()

num_ghosts = layout_obj.getNumGhosts()
state = GameState()
state.initialize(layout_obj, num_ghosts)
ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.8, 0.8)
          for i in range(num_ghosts)]

K = 10
history_states = []
history_actions = []
history_rtgs = []
rtg = 500.0
step = 0

while not (state.isWin() or state.isLose()) and step < 1000:
    feat = extract_features(state).astype(np.float32)
    history_states.append(feat)
    history_actions.append(np.zeros(5, dtype=np.float32))
    history_rtgs.append(rtg)

    # Build context
    ctx_s = np.array(history_states[-K:], dtype=np.float32)
    ctx_a = np.array(history_actions[-K:], dtype=np.float32)
    ctx_r = np.array(history_rtgs[-K:], dtype=np.float32)
    ctx_t = np.arange(len(ctx_s), dtype=np.int64)
    if len(ctx_s) < K:
        pad = K - len(ctx_s)
        ctx_s = np.pad(ctx_s, ((pad, 0), (0, 0)))
        ctx_a = np.pad(ctx_a, ((pad, 0), (0, 0)))
        ctx_r = np.pad(ctx_r, (pad, 0))
        ctx_t = np.pad(ctx_t, (pad, 0))

    with torch.no_grad():
        rtg_t = torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1)
        s_t   = torch.FloatTensor(ctx_s).unsqueeze(0)
        a_t   = torch.FloatTensor(ctx_a).unsqueeze(0)
        t_t   = torch.LongTensor(ctx_t).unsqueeze(0)
        logits, _, _ = model(rtg_t, s_t, a_t, t_t)
        all_logits = logits[0, -1, :].cpu().numpy()
        # Only consider legal actions
        legal = state.getLegalPacmanActions()
        legal_ids = [k for k, v in ACTION_MAP_STR.items() if v in legal]
        masked = {i: all_logits[i] if i in legal_ids else -float('inf') for i in range(5)}
        action_id = max(masked, key=masked.get)

    action_str = ACTION_MAP_STR[action_id]
    prev_score = state.getScore()
    state = state.generateSuccessor(0, action_str)
    reward = state.getScore() - prev_score
    rtg -= reward

    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[action_id] = 1.0
    history_actions[-1] = one_hot

    if state.isWin() or state.isLose():
        break

    for gi, ghost in enumerate(ghosts):
        if state.isWin() or state.isLose():
            break
        ga = ghost.getAction(state)
        state = state.generateSuccessor(gi + 1, ga or Directions.STOP)

    step += 1

print(f"  Final score: {state.getScore():.1f}, steps: {step}, win: {state.isWin()}")
print(f"\n===== Smoke test passed =====")
