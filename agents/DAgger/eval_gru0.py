"""Quick eval of GRU+Dueling Model 0 on mediumClassic."""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

H,W,C,SEQ=11,20,8,3
ACT={Directions.NORTH:0,Directions.SOUTH:1,Directions.EAST:2,Directions.WEST:3,Directions.STOP:4}
REV={v:k for k,v in ACT.items()}

class GRUDuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(C,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.gru=nn.GRU(64,128,batch_first=True)
        self.v_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.a_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x):
        if x.dim()==4: x=x.unsqueeze(1)
        B,T=x.shape[:2]
        feats=self.conv(x.view(B*T,C,H,W)).mean(dim=[2,3]).view(B,T,64)
        _,h=self.gru(feats)
        h=h.squeeze(0)
        v=self.v_head(h); a=self.a_head(h)
        return v+a-a.mean(dim=-1,keepdim=True)

# Model
model=GRUDuelingDQN()
model.load_state_dict(torch.load(os.path.join(PROJECT,'checkpoints/gruduel_m0_final.pt'),map_location='cpu'))
model.eval()
print(f'Loaded GRU+Dueling Model 0')

# Walls
lo=layout.getLayout('mediumClassic')
w_grid=np.zeros((H,W),dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]: w_grid[y,x]=1.0

def s2g(state):
    g=np.zeros((C,H,W),dtype=np.float32)
    food=state.getFood()
    for x in range(W):
        for y in range(H):
            if food[x][y]: g[0,y,x]=1.0
    for cx,cy in state.getCapsules():
        if 0<=cx<W and 0<=cy<H: g[1,cy,cx]=1.0
    px,py=state.getPacmanPosition()
    if 0<=px<W and 0<=py<H: g[2,py,px]=1.0
    for i,gh in enumerate(state.getGhostStates()):
        gx,gy=int(gh.getPosition()[0]),int(gh.getPosition()[1])
        if 0<=gx<W and 0<=gy<H:
            g[3+i,gy,gx]=1.0; g[5+i,gy,gx]=gh.scaredTimer/40.0
    g[7]=w_grid; return g

ghosts=[ghostAgents.DirectionalGhost(i+1,0.8,0.8) for i in range(lo.getNumGhosts())]

scores,wins=[],0
for ep in range(10):
    state=GameState(); state.initialize(lo,lo.getNumGhosts())
    # History buffer for GRU
    hist=[]
    step=0
    while not(state.isWin() or state.isLose()) and step<500:
        grid=s2g(state)
        hist.append(grid)
        if len(hist)>SEQ: hist=hist[-SEQ:]
        # Pad to SEQ if not enough history
        while len(hist)<SEQ: hist.insert(0,hist[0])

        seq=np.stack(hist)  # (SEQ, C, H, W)
        q=model(torch.FloatTensor(seq).unsqueeze(0))[0].detach().numpy()

        legal=state.getLegalActions(0)
        ids=[ACT[a] for a in legal if a!=Directions.STOP or len(legal)==1]
        if not ids: ids=[4]
        masked={i:q[i] if i in ids else -float('inf') for i in range(5)}
        state=state.generateSuccessor(0,REV[max(masked,key=masked.get)])
        if state.isWin() or state.isLose(): break
        for gi,g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state=state.generateSuccessor(gi+1,g.getAction(state) or Directions.STOP)
        step+=1
    scores.append(state.getScore())
    if state.isWin(): wins+=1
    print(f'Ep{ep:2d}: score={scores[-1]:6.0f}  win={state.isWin()}')

print(f'\nGRU+Dueling M0: Avg={np.mean(scores):.0f}  Wins={wins}/10  Min={min(scores):.0f}  Max={max(scores):.0f}')
