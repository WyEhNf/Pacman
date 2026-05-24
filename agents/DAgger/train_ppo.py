"""PPO warm-started from DAgger R1 — following pacman-ai-master approach.

4-frame stack → Actor-Critic → PPO with GAE.
"""
import sys, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

H, W, C = 11, 20, 8
N_FRAMES = 4; IN_CH = C * N_FRAMES
GAMMA = 0.99; LAMBDA = 0.95; CLIP_EPS = 0.2
ENTROPY_COEF = 0.05; VALUE_COEF = 1.0
LR = 5e-5; ROLLOUT_STEPS = 256; PPO_EPOCHS = 3; BATCH_SIZE = 64
TEMPERATURE = 5.0  # soften policy from Q-value warm-start

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}

def get_walls():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo
WG, LO = get_walls()

class PacmanEnv:
    def __init__(self, ghost_attack=0.5):
        self.ghost_attack = ghost_attack; self.reset()

    def reset(self):
        self.state = GameState(); self.state.initialize(LO, LO.getNumGhosts())
        self.ghosts = [ghostAgents.DirectionalGhost(i+1, self.ghost_attack, 0.5)
                       for i in range(LO.getNumGhosts())]
        self.frame_stack = deque(maxlen=N_FRAMES)
        g = self._grid()
        for _ in range(N_FRAMES): self.frame_stack.append(g.copy())
        return self._obs()

    def _grid(self):
        g = np.zeros((C, H, W), dtype=np.float32); st = self.state
        fd = st.getFood()
        for x in range(W):
            for y in range(H):
                if fd[x][y]: g[0, y, x] = 1.0
        for cx, cy in st.getCapsules():
            if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
        px, py = st.getPacmanPosition()
        if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0
        for i, gh in enumerate(st.getGhostStates()):
            gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
            if 0 <= gx < W and 0 <= gy < H:
                g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = gh.scaredTimer / 40.0
        for x in range(W):
            for y in range(H):
                if LO.walls.data[x][y]: g[7, y, x] = 1.0
        return g

    def _obs(self):
        return np.concatenate(list(self.frame_stack), axis=0)

    def step(self, action):
        prev = self.state.getScore()
        self.state = self.state.generateSuccessor(0, action)
        done = False
        if not (self.state.isWin() or self.state.isLose()):
            for g in self.ghosts:
                if self.state.isWin() or self.state.isLose(): break
                self.state = self.state.generateSuccessor(g.index, g.getAction(self.state) or Directions.STOP)
        reward = self.state.getScore() - prev
        done = self.state.isWin() or self.state.isLose()
        g = self._grid(); self.frame_stack.append(g.copy())
        return self._obs(), reward, done, self.state.isWin() if done else False

    def legal_actions(self):
        return [ACT[a] for a in self.state.getLegalActions(0) if a != Directions.STOP or len(self.state.getLegalActions(0)) == 1]

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_CH, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.shared_fc = nn.Linear(64, 128)
        self.policy_head = nn.Linear(128, 5)
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        f = self.conv(x).mean(dim=[2, 3]); f = F.relu(self.shared_fc(f))
        return self.policy_head(f), self.value_head(f)

    @torch.no_grad()
    def act(self, obs, legal_ids, deterministic=False, device='cpu'):
        t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits, value = self(t)
        # Mask illegal actions
        mask = torch.ones(5).to(device) * -1e9
        for lid in legal_ids: mask[lid] = 0
        logits = logits + mask
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            a = legal_ids[torch.argmax(logits[0, legal_ids]).item()]
        else:
            a = legal_ids[torch.multinomial(probs[0, legal_ids], 1).item()]
        return a, value.item(), probs[0, a].item()

def warm_start(model):
    pt = torch.load(os.path.join(PROJECT, 'checkpoints', 'dagger_cnn_m0_final.pt'), map_location='cpu')
    c0 = pt['conv.0.weight']
    model.conv[0].weight.data.copy_(c0.repeat(1, N_FRAMES, 1, 1) / N_FRAMES)
    for i in [1, 3, 5]:
        k = f'conv.{i}.weight'
        if k in pt: model.conv[i].weight.data.copy_(pt[k]); model.conv[i].bias.data.copy_(pt[f'conv.{i}.bias'])
    model.shared_fc.weight.data.copy_(pt['fc.0.weight']); model.shared_fc.bias.data.copy_(pt['fc.0.bias'])
    # Policy/value heads randomly initialized (not from Q-head)
    print('Warm-started CNN+FC from DAgger R1, heads random'); return model

def collect_rollout(env, model, n, device):
    ob, ac, rw, vl, pr, dn = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(n):
        legals = env.legal_actions()
        a, v, p = model.act(obs, legals, device=device)
        no, r, d, w = env.step(REV[a])
        ob.append(obs); ac.append(a); rw.append(r); vl.append(v); pr.append(p); dn.append(float(d))
        obs = no
        if d: obs = env.reset()
    _, lv, _ = model.act(obs, env.legal_actions(), device=device)
    return (np.array(ob), np.array(ac), np.array(rw), np.array(vl), np.array(pr),
            np.array(dn), lv)

def compute_gae(rewards, values, dones, last_val):
    T = len(rewards); adv = np.zeros(T, dtype=np.float32); gae = 0.0
    for t in range(T-1, -1, -1):
        nv = values[t+1] if t+1 < T else last_val; nd = dones[t]
        delta = rewards[t] + GAMMA * nv * (1 - nd) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - nd) * gae; adv[t] = gae
    return adv, adv + values

def ppo_update(model, opt, obs, actions, old_probs, advs, rets, device):
    model.train(); pl, vl, en = 0, 0, 0; n = len(obs)
    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(n)
        for s in range(0, n, BATCH_SIZE):
            i = idx[s:s+BATCH_SIZE]
            bo = torch.FloatTensor(obs[i]).to(device); ba = torch.LongTensor(actions[i]).to(device)
            ba_adv = torch.FloatTensor(advs[i]).to(device); ba_ret = torch.FloatTensor(rets[i]).to(device)
            ba_old = torch.FloatTensor(old_probs[i]).to(device)
            logits, values = model(bo)
            probs = F.softmax(logits, -1); np_ = probs.gather(1, ba.unsqueeze(-1)).squeeze(-1)
            ratio = np_ / (ba_old + 1e-8)
            s1 = ratio * ba_adv; s2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * ba_adv
            p_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(values.squeeze(-1), ba_ret)
            ent = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            loss = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            pl+=p_loss.item(); vl+=v_loss.item(); en+=ent.item()
    nu = PPO_EPOCHS * ((n+BATCH_SIZE-1)//BATCH_SIZE)
    return pl/nu, vl/nu, en/nu

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PPO on {device} | DAgger R1 warm-start | frame_stack={N_FRAMES}\n')

    model = warm_start(ActorCritic()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = PacmanEnv(ghost_attack=0.5)
    ep_rew = []; best = -999

    for it in range(1000):
        obs, acts, rews, vals, probs, dones, lv = collect_rollout(env, model, ROLLOUT_STEPS, device)
        advs, rets = compute_gae(rews, vals, dones, lv)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        pl, vl, ent = ppo_update(model, opt, obs, acts, probs, advs, rets, device)

        cr = 0
        for r, d in zip(rews, dones):
            cr += r
            if d: ep_rew.append(cr); cr = 0

        if it % 10 == 0:
            r50 = ep_rew[-50:] if len(ep_rew)>=50 else ep_rew
            avg = np.mean(r50) if r50 else 0; w = sum(1 for r in r50 if r>900)
            print(f'Iter{it:5d} | p={pl:.4f} v={vl:.3f} ent={ent:.3f} | avg50={avg:.0f} wins={w}')
            if avg > best: best = avg; torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints', 'ppo_best.pt'))

    torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints', 'ppo_final.pt'))

if __name__ == '__main__':
    main()
