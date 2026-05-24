"""Quick eval of warm-start DQN on Pacman."""
import sys, os, numpy as np, torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState
from scripts.collect_expert_data import extract_features
import torch.nn as nn
class EvalDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(448, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 5)
    def forward(self, x):
        return self.q_head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

model = EvalDQN()
model.load_state_dict(torch.load(os.path.join(PROJECT, 'checkpoints/dqn_selfplay_best.pt'), map_location='cpu'))
model.eval()

def pad(f):
    if len(f) == 448: return f.astype(np.float32)
    p = np.zeros(448, np.float32); p[:len(f)] = f; return p

lo = layout.getLayout('mediumClassic')
ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]
scores, wins = [], 0

for ep in range(20):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        feat = pad(extract_features(state))
        q = model(torch.FloatTensor(feat).unsqueeze(0))[0].detach().numpy()
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        masked = {i: q[i] if i in ids else -float('inf') for i in range(5)}
        aid = max(masked, key=masked.get)
        state = state.generateSuccessor(0, REV[aid])
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        step += 1
    scores.append(state.getScore())
    if state.isWin(): wins += 1
    print(f'Ep{ep:2d}: score={scores[-1]:6.0f}  win={state.isWin()}')

print(f'\nAvg: {np.mean(scores):.0f}  Wins: {wins}/20')
