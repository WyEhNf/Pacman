"""
Pure DQN self-play training with experience replay.
No MCTS — just epsilon-greedy exploration + Bellman updates.

Usage:
    python scripts/train_dqn_selfplay.py --episodes 200 --warmstart checkpoints/dqn_warmstart.pt
"""

import sys, os, time, random, argparse
from collections import deque
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL); sys.path.insert(0, PROJECT)
os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState
from scripts.collect_expert_data import extract_features

ACT_MAP  = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
ACT_REV  = {v: k for k, v in ACT_MAP.items()}
STATE_DIM = 448


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 5)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.q_head(h)

def pad_feat(f):
    if len(f) == STATE_DIM: return f.astype(np.float32)
    p = np.zeros(STATE_DIM, np.float32); p[:len(f)] = f; return p

def play_episode(model, lo, ghosts, epsilon, device='cpu'):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    buffer, step = [], 0
    while not (state.isWin() or state.isLose()) and step < 500:
        feat = pad_feat(extract_features(state))
        legal = state.getLegalActions(0)
        ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]

        if np.random.random() < epsilon:
            aid = np.random.choice(ids)
        else:
            with torch.no_grad():
                q = model(torch.FloatTensor(feat).unsqueeze(0).to(device))[0].cpu().numpy()
            masked = {i: q[i] if i in ids else -float('inf') for i in range(5)}
            aid = max(masked, key=masked.get)

        prev_score = state.getScore()
        state = state.generateSuccessor(0, ACT_REV[aid])
        reward = state.getScore() - prev_score

        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)

        next_feat = pad_feat(extract_features(state))
        buffer.append((feat, aid, reward, next_feat, state.isWin() or state.isLose()))
        step += 1
        if state.isWin() or state.isLose(): break
    return buffer, state.getScore(), state.isWin()

def train_batch(model, opt, batch, expert_mask, device, gamma=0.99, ce_weight=0.5):
    """Mixed training: CE (supervised) for expert data, TD for self-play."""
    s  = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
    a  = torch.LongTensor(np.array([b[1] for b in batch])).to(device)
    r  = torch.FloatTensor(np.array([b[2] for b in batch])).to(device)
    ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
    d  = torch.FloatTensor(np.array([b[4] for b in batch])).to(device)

    q_all = model(s)
    total_loss = 0.0

    # --- TD loss on ALL data ---
    r_clip = torch.clamp(r, -100, 100) / 100.0
    q_pred = q_all[range(len(batch)), a]
    with torch.no_grad():
        q_next = model(ns)
        q_target = r_clip + gamma * (1 - d) * q_next.max(dim=-1).values
        q_target = torch.clamp(q_target, -5, 5)
    td_loss = F.mse_loss(q_pred, q_target)
    total_loss += td_loss

    # --- CE loss on EXPERT data only (preserve warmstart knowledge) ---
    if expert_mask.sum() > 0:
        expert_logits = q_all[expert_mask]
        expert_actions = a[expert_mask]
        ce_loss = F.cross_entropy(expert_logits, expert_actions)
        total_loss += ce_weight * ce_loss

    opt.zero_grad(); total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return total_loss.item()

def evaluate(model, lo, ghosts, n_games, device):
    wins = 0; scores = []
    for _ in range(n_games):
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        while not (state.isWin() or state.isLose()):
            feat = pad_feat(extract_features(state))
            with torch.no_grad():
                q = model(torch.FloatTensor(feat).unsqueeze(0).to(device))[0].cpu().numpy()
            legal = state.getLegalActions(0)
            ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
            if not ids: ids = [4]
            masked = {i: q[i] if i in ids else -float('inf') for i in range(5)}
            aid = max(masked, key=masked.get)
            state = state.generateSuccessor(0, ACT_REV[aid])
            for gi, g in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
            if state.isWin() or state.isLose(): break
        scores.append(state.getScore())
        if state.isWin(): wins += 1
    return np.mean(scores), wins / n_games

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--warmstart', default='checkpoints/dqn_warmstart.pt')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--epsilon_start', type=float, default=0.5)
    p.add_argument('--epsilon_end', type=float, default=0.05)
    p.add_argument('--replay_size', type=int, default=20000)
    p.add_argument('--train_every', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Episodes: {args.episodes}')

    model = DQN().to(device)
    if args.warmstart:
        ws = args.warmstart if os.path.isabs(args.warmstart) else os.path.join(PROJECT, args.warmstart)
        ckpt = torch.load(ws, map_location=device)
        # Handle full DualHeadDQN checkpoint (ignore v_head)
        sd = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        model.load_state_dict(sd, strict=True)
        print(f'Loaded warmstart: {ws}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)
    expert_buffer = []  # fixed expert wins — never evicted, 50% of each batch

    # Pre-load expert wins into immortal buffer
    wins_path = os.path.join(PROJECT, 'data/wins_only.npz')
    if os.path.exists(wins_path):
        wd = np.load(wins_path, allow_pickle=True)
        for t in wd['trajectories']:
            s, a, r = t['states'], t['actions'], t['rewards']
            for i in range(len(a)-1):
                if s.shape[1] < STATE_DIM:
                    p = np.zeros((s.shape[0], STATE_DIM), np.float32)
                    p[:,:s.shape[1]] = s; s = p
                expert_buffer.append((s[i].astype(np.float32), a[i+1], r[i+1],
                                      s[i+1].astype(np.float32), False))
        print(f'Pre-loaded {len(expert_buffer)} expert transitions (immortal)')

    lo = layout.getLayout('mediumClassic')
    ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]
    scores, wins = [], 0
    best_wr, best_avg = 0, -float('inf')

    for ep in range(args.episodes):
        eps = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * ep / args.episodes
        t0 = time.time()
        buf, score, win = play_episode(model, lo, ghosts, eps, device)
        replay.extend(buf)
        scores.append(score)
        if win: wins += 1

        status = f'[{ep+1:3d}/{args.episodes}] score={score:7.0f}  eps={eps:.2f}'
        if len(scores) >= 10:
            status += f'  avg10={np.mean(scores[-10:]):7.0f}'
        status += f'  wr={wins/(ep+1)*100:.1f}%  buffer={len(replay)}  [{time.time()-t0:.0f}s]'
        print(status)

        if len(replay) >= args.batch_size // 2 and ep % args.train_every == 0:
            # 50:50 mix — expert (never diluted) + self-play
            n_expert = min(args.batch_size // 2, len(expert_buffer))
            n_self = min(args.batch_size // 2, len(replay))
            expert_batch = random.sample(expert_buffer, n_expert)
            self_batch = random.sample(replay, n_self)
            batch = expert_batch + self_batch
            # Mark which entries are expert (first n_expert)
            expert_mask = torch.zeros(len(batch), dtype=torch.bool)
            expert_mask[:n_expert] = True
            loss = train_batch(model, opt, batch, expert_mask, device)
            print(f'      train loss={loss:.4f}  (expert:{n_expert} self:{n_self})')

        if (ep+1) % 20 == 0:
            avg_s, wr = evaluate(model, lo, ghosts, 10, device)
            print(f'      EVAL: avg_score={avg_s:.0f}  win_rate={wr:.2f}')
            if avg_s > best_avg or wr > best_wr:
                best_avg = max(best_avg, avg_s); best_wr = max(best_wr, wr)
                torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints/dqn_selfplay_best.pt'))

    torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints/dqn_selfplay_final.pt'))
    avg_s, wr = evaluate(model, lo, ghosts, 20, device)
    print(f'\nFinal: avg={avg_s:.0f}  wr={wr:.2f}  best: avg={best_avg:.0f} wr={best_wr:.2f}')

if __name__ == '__main__':
    main()
