"""Quick test: load DT checkpoint and play one game with rendering."""
import sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL)
os.chdir(SKEL)

import torch, numpy as np
import layout, ghostAgents
from game import Directions
from pacman import GameState
from src.model.decision_transformer import DecisionTransformer
from scripts.collect_expert_data import extract_features

ACTION_MAP = {0: Directions.NORTH, 1: Directions.SOUTH, 2: Directions.EAST, 3: Directions.WEST, 4: Directions.STOP}

ckpt = torch.load('e:/Pacman/checkpoints/dt_v1_100ep.pt', map_location='cpu')
state_dim = ckpt['state_dim']

model = DecisionTransformer(state_dim=state_dim, act_dim=5, d_model=256, n_heads=4, n_layers=4, context_len=20)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded DT_v1 (100 eps)  |  state_dim={state_dim}  rtg_range=[{ckpt["rtg_min"]:.0f}, {ckpt["rtg_max"]:.0f}]')

# Pick layout
lo = layout.getLayout('mediumClassic')
ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]
state = GameState(); state.initialize(lo, lo.getNumGhosts())

K = 20
h_states, h_actions, h_rtgs = [], [], []
rtg = ckpt['rtg_max'] * 0.7  # target a high score
step = 0

while not (state.isWin() or state.isLose()) and step < 500:
    feat = extract_features(state).astype(np.float32)
    if len(feat) != state_dim:
        # pad to model's state_dim
        padded = np.zeros(state_dim, dtype=np.float32)
        padded[:len(feat)] = feat; feat = padded
    h_states.append(feat); h_actions.append(np.zeros(5, np.float32)); h_rtgs.append(rtg)

    ctx_s = np.array(h_states[-K:], np.float32); ctx_a = np.array(h_actions[-K:], np.float32)
    ctx_r = np.array(h_rtgs[-K:], np.float32); ctx_t = np.arange(len(ctx_s), dtype=np.int64)
    if len(ctx_s) < K:
        p = K - len(ctx_s)
        ctx_s = np.pad(ctx_s, ((p,0),(0,0))); ctx_a = np.pad(ctx_a, ((p,0),(0,0)))
        ctx_r = np.pad(ctx_r, (p,0)); ctx_t = np.pad(ctx_t, (p,0))

    with torch.no_grad():
        logits, _, _ = model(
            torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1),
            torch.FloatTensor(ctx_s).unsqueeze(0),
            torch.FloatTensor(ctx_a).unsqueeze(0),
            torch.LongTensor(ctx_t).unsqueeze(0))
        # Filter to legal actions
        legal = state.getLegalActions(0)
        l = logits[0,-1,:].numpy()
        masked = {i: l[i] if ACTION_MAP[i] in legal else -float('inf') for i in range(5)}
        aid = max(masked, key=masked.get)

    astr = ACTION_MAP[aid]
    prev_score = state.getScore()
    state = state.generateSuccessor(0, astr)
    reward = state.getScore() - prev_score
    rtg -= reward
    h_actions[-1][aid] = 1.0
    step += 1

    if state.isWin() or state.isLose(): break
    for gi, g in enumerate(ghosts):
        if state.isWin() or state.isLose(): break
        ga = g.getAction(state)
        state = state.generateSuccessor(gi+1, ga or Directions.STOP)

print(f'Score: {state.getScore():.0f}  |  Steps: {step}  |  Win: {state.isWin()}')

# Run 10 episodes silently
scores = []
for ep in range(10):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    h_states, h_actions, h_rtgs = [], [], []
    rtg = ckpt['rtg_max'] * 0.7; step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        feat = extract_features(state).astype(np.float32)
        if len(feat) != state_dim:
            padded = np.zeros(state_dim, np.float32); padded[:len(feat)] = feat; feat = padded
        h_states.append(feat); h_actions.append(np.zeros(5, np.float32)); h_rtgs.append(rtg)
        ctx_s = np.array(h_states[-K:], np.float32); ctx_a = np.array(h_actions[-K:], np.float32)
        ctx_r = np.array(h_rtgs[-K:], np.float32); ctx_t = np.arange(len(ctx_s), dtype=np.int64)
        if len(ctx_s) < K:
            p = K - len(ctx_s); ctx_s = np.pad(ctx_s, ((p,0),(0,0)))
            ctx_a = np.pad(ctx_a, ((p,0),(0,0))); ctx_r = np.pad(ctx_r, (p,0)); ctx_t = np.pad(ctx_t, (p,0))
        with torch.no_grad():
            logits, _, _ = model(torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1),
                                 torch.FloatTensor(ctx_s).unsqueeze(0),
                                 torch.FloatTensor(ctx_a).unsqueeze(0), torch.LongTensor(ctx_t).unsqueeze(0))
            legal = state.getLegalActions(0)
            l = logits[0,-1,:].numpy()
            masked = {i: l[i] if ACTION_MAP[i] in legal else -float('inf') for i in range(5)}
            aid = max(masked, key=masked.get)
        astr = ACTION_MAP[aid]
        prev = state.getScore(); state = state.generateSuccessor(0, astr)
        rtg -= state.getScore() - prev; h_actions[-1][aid] = 1.0; step += 1
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
    scores.append(state.getScore())
print(f'\n10 episodes on mediumClassic:')
print(f'  Scores: {[f"{s:.0f}" for s in scores]}')
print(f'  Avg: {np.mean(scores):.0f}  |  Wins: {sum(1 for s in scores if s > 0)}/10')
