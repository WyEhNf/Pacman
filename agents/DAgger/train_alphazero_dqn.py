"""
AlphaZero-style MCTS + DQN self-play training.

MCTS self-plays to generate data → trains DualHeadDQN → DQN guides MCTS → loop.

Usage:
    python scripts/train_alphazero_dqn.py --iterations 20
"""

import sys, os, time, math, random, argparse
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
STATE_DIM = 448  # fixed for mediumClassic (max layout)


# ═══════════════════════════════════════════════════════════════════════
#  Dual-Head DQN
# ═══════════════════════════════════════════════════════════════════════

class DualHeadDQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, act_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, act_dim)   # Q(s, a) per action
        self.v_head = nn.Linear(128, 1)          # V(s) scalar

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.q_head(h), self.v_head(h)


# ═══════════════════════════════════════════════════════════════════════
#  MCTS with DQN prior
# ═══════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ('N', 'W', 'Q', 'P', 'children', 'parent')
    def __init__(self, prior=0.0):
        self.N = 0; self.W = 0.0; self.Q = 0.0; self.P = prior
        self.children = {}; self.parent = None


class MCTSDQN:
    def __init__(self, n_sim=200, c_puct=2.0, tau=1.0):
        self.n_sim = n_sim; self.c_puct = c_puct; self.tau = tau

    def search(self, state_feat, model, legal_ids, env_state, ghosts, device='cpu'):
        """Run MCTS with real-game rollout for leaf evaluation."""
        root = MCTSNode()
        s_tensor = torch.FloatTensor(state_feat).unsqueeze(0).to(device)

        # Get DQN priors for root
        with torch.no_grad():
            q_vals, _ = model(s_tensor)
            q_np = q_vals[0].cpu().numpy()
        priors = np.exp(np.clip(q_np, -10, 10) / self.tau)
        for a in range(5):
            if a not in legal_ids: priors[a] = 0.0
        priors /= priors.sum() + 1e-8

        for a in legal_ids:
            child = MCTSNode(prior=priors[a])
            child.parent = root
            root.children[a] = child

        for _ in range(self.n_sim):
            # Select
            node = root; path = []
            while node.children:
                best_u = -float('inf'); best_a = None
                for a, c in node.children.items():
                    u = c.Q + self.c_puct * c.P * math.sqrt(node.N + 1) / (1 + c.N)
                    if u > best_u: best_u = u; best_a = a
                node = node.children[best_a]; path.append(best_a)

            # Leaf evaluation: random rollout (generateSuccessor is non-destructive)
            value = self._rollout(env_state, ghosts)

            for a in path:
                root.children[a].N += 1
                root.children[a].W += value
                root.children[a].Q = root.children[a].W / root.children[a].N
            root.N += 1; root.W += value

        visits = np.zeros(5)
        for a, c in root.children.items():
            visits[a] = c.N
        visits /= visits.sum() + 1e-8
        return visits, root

    def _rollout(self, state, ghosts, depth=8):
        """Random rollout from a game state. Returns final score."""
        for _ in range(depth):
            if state.isWin() or state.isLose(): break
            legal = state.getLegalActions(0)
            a = random.choice(legal)
            state = state.generateSuccessor(0, a)
            if state.isWin() or state.isLose(): break
            for gi, g in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        return state.getScore()


# ═══════════════════════════════════════════════════════════════════════
#  Environment helpers
# ═══════════════════════════════════════════════════════════════════════

def get_state_dim(lo):
    return 2 + 2*lo.getNumGhosts() + lo.getNumGhosts() + lo.width*lo.height + lo.width*lo.height

def pad_feat(feat, dim=STATE_DIM):
    if len(feat) == dim: return feat.astype(np.float32)
    p = np.zeros(dim, np.float32); p[:len(feat)] = feat; return p

def create_env(layout_name):
    lo = layout.getLayout(layout_name)
    ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]
    return lo, ghosts

def reset_env(lo, ghosts):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    return state

def step_env(state, action_str, ghosts):
    """Execute Pacman + ghost turns. Returns (next_state, reward, done)."""
    prev = state.getScore()
    state = state.generateSuccessor(0, action_str)
    reward = state.getScore() - prev
    if state.isWin() or state.isLose(): return state, reward, True
    for gi, g in enumerate(ghosts):
        if state.isWin() or state.isLose(): break
        state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
    done = state.isWin() or state.isLose()
    return state, reward, done


# ═══════════════════════════════════════════════════════════════════════
#  Self-play
# ═══════════════════════════════════════════════════════════════════════

def selfplay_episode(mcts, model, lo, ghosts, device):
    """One MCTS self-play episode. Returns list of transitions."""
    state = reset_env(lo, ghosts)
    buffer = []
    prev_score = state.getScore(); step = 0

    while not (state.isWin() or state.isLose()) and step < 500:
        feat = pad_feat(extract_features(state))
        legal = state.getLegalActions(0)
        legal_ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]

        if len(legal_ids) == 0:
            legal_ids = [4]  # STOP

        visit_dist, root = mcts.search(feat, model, legal_ids, state, ghosts, device)
        # Sample action from visit distribution (temperature sampling)
        if np.random.random() < 0.25:
            aid = np.random.choice(legal_ids)  # explore
        else:
            dist = visit_dist[legal_ids] / visit_dist[legal_ids].sum()
            aid = np.random.choice(legal_ids, p=dist)

        astr = ACT_REV[aid]
        state, reward, done = step_env(state, astr, ghosts)
        next_feat = pad_feat(extract_features(state))

        buffer.append({
            'state': feat, 'mcts_policy': visit_dist, 'action': aid,
            'reward': reward, 'next_state': next_feat, 'done': done
        })
        step += 1

        if done: break

    final_score = state.getScore()
    return buffer, final_score, state.isWin()


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def train_step(model, opt, batch, device, gamma=0.99, policy_weight=0.3):
    states   = torch.FloatTensor(np.array([b['state'] for b in batch])).to(device)
    mcts_pi  = torch.FloatTensor(np.array([b['mcts_policy'] for b in batch])).to(device)
    actions  = torch.LongTensor(np.array([b['action'] for b in batch])).to(device)
    rewards  = torch.FloatTensor(np.array([b['reward'] for b in batch])).to(device)
    next_s   = torch.FloatTensor(np.array([b['next_state'] for b in batch])).to(device)
    dones    = torch.FloatTensor(np.array([b['done'] for b in batch])).to(device)

    q_vals, v_vals = model(states)
    q_pred = q_vals[range(len(batch)), actions]

    with torch.no_grad():
        q_next, _ = model(next_s)
    # Normalize: clip rewards to [-100, 100] then scale to [-1, 1]
    rewards = torch.clamp(rewards, -100, 100) / 100.0
    q_target = rewards + gamma * (1 - dones) * q_next.max(dim=-1).values
    q_target = torch.clamp(q_target, -5, 5)  # prevent Q explosion

    td_loss = F.mse_loss(q_pred, q_target)

    # Policy loss: KL between MCTS visit distribution and softmax(Q)
    log_probs = F.log_softmax(q_vals / 1.0, dim=-1)
    policy_loss = -(mcts_pi * log_probs).sum(dim=-1).mean()

    loss = td_loss + policy_weight * policy_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return {'td': td_loss.item(), 'policy': policy_loss.item(), 'total': loss.item()}


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, lo, ghosts, device, n_games=10):
    """Play N games using DQN greedy policy (no MCTS)."""
    wins = 0; scores = []
    for _ in range(n_games):
        state = reset_env(lo, ghosts)
        while not (state.isWin() or state.isLose()):
            feat = pad_feat(extract_features(state))
            s_t = torch.FloatTensor(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                q, _ = model(s_t); q = q[0].cpu().numpy()
            legal = state.getLegalActions(0)
            legal_ids = [ACT_MAP[a] for a in legal if a != Directions.STOP or len(legal) == 1]
            if not legal_ids: legal_ids = [4]
            masked = {i: q[i] if i in legal_ids else -float('inf') for i in range(5)}
            aid = max(masked, key=masked.get)
            state, _, done = step_env(state, ACT_REV[aid], ghosts)
            if done: break
        scores.append(state.getScore())
        if state.isWin(): wins += 1
    return np.mean(scores), wins / n_games


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--episodes_per_iter', type=int, default=5)
    p.add_argument('--mcts_sims', type=int, default=150)
    p.add_argument('--train_epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--replay_size', type=int, default=10000)
    p.add_argument('--warmstart', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  MCTS sims: {args.mcts_sims}')
    print(f'Iterations: {args.iterations}  |  Ep/iter: {args.episodes_per_iter}')

    model = DualHeadDQN().to(device)
    if args.warmstart:
        ws_path = args.warmstart if os.path.isabs(args.warmstart) else os.path.join(PROJECT, args.warmstart)
        model.load_state_dict(torch.load(ws_path, map_location=device))
        print(f'Loaded warmstart: {ws_path}')
    mcts = MCTSDQN(n_sim=args.mcts_sims)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)

    lo, ghosts = create_env('mediumClassic')
    print(f'Layout: mediumClassic  |  Ghosts: {lo.getNumGhosts()}  |  State dim: {STATE_DIM}')

    best_wr = 0.0

    for it in range(args.iterations):
        # --- Self-play ---
        t0 = time.time()
        ep_data = []
        for ep in range(args.episodes_per_iter):
            buf, score, win = selfplay_episode(mcts, model, lo, ghosts, device)
            replay.extend(buf)
            ep_data.append((score, win))
        dt_sp = time.time() - t0

        scores = [s for s, w in ep_data]
        wins   = sum(1 for s, w in ep_data if w)
        print(f'[{it+1:3d}] selfplay: {len(ep_data)}eps  '
              f'avg_score={np.mean(scores):.0f}  wins={wins}  [{dt_sp:.0f}s]')

        # --- Train ---
        if len(replay) < args.batch_size:
            continue
        t0 = time.time()
        losses = []
        for _ in range(args.train_epochs):
            batch = random.sample(replay, min(args.batch_size, len(replay)))
            loss_dict = train_step(model, opt, batch, device)
            losses.append(loss_dict)
        dt_tr = time.time() - t0

        td = np.mean([l['td'] for l in losses])
        pl = np.mean([l['policy'] for l in losses])
        print(f'      train: td_loss={td:.4f}  policy_loss={pl:.4f}  '
              f'buffer={len(replay)}  [{dt_tr:.0f}s]')

        # --- Evaluate ---
        if (it + 1) % 5 == 0:
            avg_s, wr = evaluate(model, lo, ghosts, device, n_games=10)
            print(f'      eval: avg_score={avg_s:.0f}  win_rate={wr:.2f}')
            if wr > best_wr:
                best_wr = wr
                torch.save(model.state_dict(),
                           os.path.join(PROJECT, 'checkpoints/az_dqn_best.pt'))
                print(f'      saved best model (wr={wr:.2f})')

    # Final
    avg_s, wr = evaluate(model, lo, ghosts, device, n_games=20)
    print(f'\nFinal: avg_score={avg_s:.0f}  win_rate={wr:.2f}  best_wr={best_wr:.2f}')
    torch.save(model.state_dict(), os.path.join(PROJECT, 'checkpoints/az_dqn_final.pt'))


if __name__ == '__main__':
    main()
