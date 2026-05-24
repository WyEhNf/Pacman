"""Hybrid Ensemble: CNN v5 (5 models) + GRU+Dueling best 3 (M0,M1,M2)"""
import sys,os,numpy as np,torch,torch.nn as nn
PROJECT=r'E:\Pacman'
SKEL=os.path.join(PROJECT,'PPCA-AIPacMan-2024-main','multiagent')
sys.path.insert(0,PROJECT);sys.path.insert(0,SKEL);os.chdir(SKEL)
import layout,ghostAgents
from game import Directions
from pacman import GameState

H,W,C,SEQ=11,20,8,3
ACT={Directions.NORTH:0,Directions.SOUTH:1,Directions.EAST:2,Directions.WEST:3,Directions.STOP:4}
REV={v:k for k,v in ACT.items()}

class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(C,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.fc=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,5))
    def forward(self,x): return self.fc(self.conv(x).mean(dim=[2,3]))

class GRUDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(C,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.gru=nn.GRU(64,128,batch_first=True)
        self.v_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.a_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x):
        if x.dim()==4:x=x.unsqueeze(1)
        B,T=x.shape[:2]
        f=self.conv(x.view(B*T,C,H,W)).mean(dim=[2,3]).view(B,T,64)
        _,h=self.gru(f);h=h.squeeze(0)
        v,a=self.v_head(h),self.a_head(h)
        return v+a-a.mean(dim=-1,keepdim=True)

lo=layout.getLayout('mediumClassic')
wg=np.zeros((H,W),dtype=np.float32)
for x in range(W):
    for y in range(H):
        if lo.walls.data[x][y]:wg[y,x]=1.0

def s2g(s):
    g=np.zeros((C,H,W),dtype=np.float32)
    fd=s.getFood()
    for x in range(W):
        for y in range(H):
            if fd[x][y]:g[0,y,x]=1.0
    for cx,cy in s.getCapsules():
        if 0<=cx<W and 0<=cy<H:g[1,cy,cx]=1.0
    px,py=s.getPacmanPosition()
    if 0<=px<W and 0<=py<H:g[2,py,px]=1.0
    for i,gh in enumerate(s.getGhostStates()):
        gx,gy=int(gh.getPosition()[0]),int(gh.getPosition()[1])
        if 0<=gx<W and 0<=gy<H:
            g[3+i,gy,gx]=1.0;g[5+i,gy,gx]=gh.scaredTimer/40.0
    g[7]=wg;return g

# Load CNN v5 (5 models)
cnns=[]
for i in range(5):
    m=CNNDQN();m.load_state_dict(torch.load(os.path.join(PROJECT,f'checkpoints/v5_cnn_m{i}_final.pt'),map_location='cpu'))
    m.eval();cnns.append(m)

# Load GRU+Dueling M0,M1,M2
grus=[]
for i in [0,1,2]:
    m=GRUDQN();m.load_state_dict(torch.load(os.path.join(PROJECT,f'checkpoints/gruduel_m{i}_final.pt'),map_location='cpu'))
    m.eval();grus.append(m)

print(f'Hybrid: {len(cnns)} CNN v5 + {len(grus)} GRU = {len(cnns)+len(grus)} models')

ghosts=[ghostAgents.DirectionalGhost(i+1,0.8,0.8) for i in range(lo.getNumGhosts())]
scores,wins=[],0
for ep in range(20):
    st=GameState();st.initialize(lo,lo.getNumGhosts())
    hist,step=[],0
    while not(st.isWin() or st.isLose()) and step<500:
        g=s2g(st)
        hist.append(g)
        if len(hist)>SEQ:hist=hist[-SEQ:]
        while len(hist)<SEQ:hist.insert(0,hist[0])

        t_cnn=torch.FloatTensor(g).unsqueeze(0)            # single frame
        t_gru=torch.FloatTensor(np.stack(hist)).unsqueeze(0) # sequence

        q_cnn=sum(m(t_cnn)[0].detach().numpy() for m in cnns)/len(cnns)
        q_gru=sum(m(t_gru)[0].detach().numpy() for m in grus)/len(grus)
        q=(q_cnn+q_gru)/2  # equal weight per family

        legal=st.getLegalActions(0)
        ids=[ACT[a] for a in legal if a!=Directions.STOP or len(legal)==1]
        if not ids:ids=[4]
        best,mv=-1e9,4
        for i in range(5):
            if i in ids and q[i]>best:best=q[i];mv=i
        st=st.generateSuccessor(0,REV[mv])
        if st.isWin() or st.isLose():break
        for gi,g in enumerate(ghosts):
            if st.isWin() or st.isLose():break
            st=st.generateSuccessor(gi+1,g.getAction(st) or Directions.STOP)
        step+=1
    scores.append(st.getScore())
    if st.isWin():wins+=1
    print(f'Ep{ep:2d}: {scores[-1]:6.0f}  {"WIN" if st.isWin() else ""}')

print(f'\nHybrid (CNNv5x5 + GRUx3): Avg={np.mean(scores):.0f}  Wins={wins}/20  Min={min(scores):.0f}  Max={max(scores):.0f}')
