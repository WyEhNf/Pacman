"""
Online Decision Transformer — self-play collection + continuous training.

Loads DT_v1, plays games, collects trajectories, trains online.
Mixes expert data + self-play data in the replay buffer.

Usage:
    python scripts/online_dt.py --episodes 100
"""

import sys, os, time, argparse
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL); sys.path.insert(0, PROJECT)
os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState
from src.model.decision_transformer import DecisionTransformer
from scripts.collect_expert_data import extract_features

ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
ACT_REV = {v: k for k, v in ACT_MAP.items()}


class OnlineDT:
    """DT that plays and learns from its own experience."""

    def __init__(self, model, state_dim, device, context_len=20, target_rtg=None):
        self.model = model
        self.state_dim = state_dim
        self.device = device
        self.K = context_len
        self.target_rtg = target_rtg or 500.0
        self.replay_buffer = []  # list of trajectory dicts

    def play_episode(self, layout_obj, epsilon=0.1):
        """Play one episode with epsilon-greedy exploration. Returns trajectory."""
        ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8)
                  for i in range(layout_obj.getNumGhosts())]
        state = GameState(); state.initialize(layout_obj, layout_obj.getNumGhosts())

        h_states, h_acts, h_rews = [], [], []
        rtg = self.target_rtg; step = 0; prev_score = state.getScore()

        while not (state.isWin() or state.isLose()) and step < 800:
            feat = self._pad_feat(extract_features(state))
            h_states.append(feat); h_acts.append(-1); h_rews.append(0.0)

            # Build context
            ctx_s, ctx_a, ctx_r, ctx_t, n_real = self._make_context(h_states, h_acts, h_rews)

            # Epsilon-greedy
            if np.random.random() < epsilon:
                legal = state.getLegalActions(0)
                aid = ACT_MAP[np.random.choice(legal)]
            else:
                with torch.no_grad():
                    r_t = torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1).to(self.device)
                    s_t = torch.FloatTensor(ctx_s).unsqueeze(0).to(self.device)
                    a_t = torch.FloatTensor(ctx_a).unsqueeze(0).to(self.device)
                    t_t = torch.LongTensor(ctx_t).unsqueeze(0).to(self.device)
                    logits, _, _ = self.model(r_t, s_t, a_t, t_t)
                    l = logits[0, -1, :].cpu().numpy()
                    legal = state.getLegalActions(0)
                    masked = {i: l[i] if ACT_REV.get(i) in legal else -float('inf') for i in range(5)}
                    aid = max(masked, key=masked.get)

            astr = ACT_REV.get(aid, Directions.STOP)
            state = state.generateSuccessor(0, astr)
            cur_score = state.getScore(); reward = cur_score - prev_score; prev_score = cur_score
            rtg -= reward; h_acts[-1] = aid; h_rews[-1] = reward; step += 1

            if state.isWin() or state.isLose(): break
            for gi, g in enumerate(ghosts):
                if state.isWin() or state.isLose(): break
                state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)

        if len(h_states) < 5: return None
        T = len(h_states)
        s = np.array(h_states, np.float32); a = [x for x in h_acts if x >= 0]
        r = np.array(h_rews[:len(a)], np.float32)
        s = s[:len(a)]; a = np.array(a, np.int32)
        rtg_arr = np.zeros(len(r), np.float32); run = 0.0
        for t in reversed(range(len(r))): run += r[t]; rtg_arr[t] = run
        return {'states': s, 'actions': a, 'rewards': r, 'returns_to_go': rtg_arr,
                'steps': len(a), 'score': state.getScore(), 'win': state.isWin(),
                'source': 'selfplay', 'quality_weight': 1.0 if state.isWin() else 0.3}

    def _pad_feat(self, feat):
        if len(feat) == self.state_dim: return feat.astype(np.float32)
        p = np.zeros(self.state_dim, np.float32); p[:len(feat)] = feat; return p

    def _make_context(self, h_states, h_acts, h_rews):
        K = self.K
        # Use real steps (where act >= 0)
        valid = [i for i, a in enumerate(h_acts) if a >= 0]
        n = len(valid)
        if n == 0:
            # First step: no history, context is all zeros + target rtg
            return (np.zeros((K, self.state_dim), np.float32),
                    np.zeros((K, 5), np.float32),
                    np.full(K, self.target_rtg, dtype=np.float32),
                    np.zeros(K, dtype=np.int64), 0)

        ctx_s = np.array([h_states[i] for i in valid[-K:]], np.float32)
        ctx_a = np.zeros((len(ctx_s), 5), np.float32)
        for j, i in enumerate(valid[-K:]):
            if h_acts[i] >= 0: ctx_a[j, h_acts[i]] = 1.0
        # RtG: sum of remaining rewards from each step
        total_r = sum(h_rews)
        rtg_vals = [self.target_rtg - sum(h_rews[:i]) for i in valid[-K:]]
        ctx_r = np.array(rtg_vals[-K:], np.float32)
        ctx_t = np.arange(len(ctx_s), dtype=np.int64)

        if len(ctx_s) < K:
            pad = K - len(ctx_s)
            ctx_s = np.pad(ctx_s, ((pad,0),(0,0))); ctx_a = np.pad(ctx_a, ((pad,0),(0,0)))
            ctx_r = np.pad(ctx_r, (pad,0), constant_values=self.target_rtg)
            ctx_t = np.pad(np.arange(len(ctx_t)), (pad,0))

        return ctx_s, ctx_a, ctx_r, ctx_t, n

    def train_step(self, batch_rtg, batch_states, batch_actions, batch_tsteps, batch_mask, optimizer):
        """One gradient step on a batch."""
        self.model.train()
        logits, _, _ = self.model(batch_rtg, batch_states, batch_actions, batch_tsteps)
        # Convert one-hot actions to int IDs for CE loss
        target_ids = batch_actions.argmax(dim=-1)  # (B, K)
        loss = torch.nn.functional.cross_entropy(
            logits[batch_mask.bool()].view(-1, 5),
            target_ids[batch_mask.bool()].view(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/dt_v1_100ep.pt')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--train_every', type=int, default=5, help='Train every N episodes')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--epsilon', type=float, default=0.15)
    p.add_argument('--buffer_size', type=int, default=500)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Episodes: {args.episodes}')

    # Load model
    ckpt = torch.load(os.path.join(PROJECT, args.checkpoint), map_location=device)
    sd = ckpt['state_dim']
    dt = DecisionTransformer(state_dim=sd, act_dim=5, d_model=256, n_heads=4, n_layers=4, context_len=20).to(device)
    dt.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded: {args.checkpoint}  |  state_dim={sd}')

    target_rtg = ckpt['rtg_max'] * 0.8  # aim high
    agent = OnlineDT(dt, sd, device, target_rtg=target_rtg)
    opt = torch.optim.AdamW(dt.parameters(), lr=args.lr)

    # Load expert data as initial buffer
    expert_path = os.path.join(PROJECT, 'data/inc_0220.npz')
    if os.path.exists(expert_path):
        ed = np.load(expert_path, allow_pickle=True)
        raw = list(ed['trajectories'])
        # Pad expert states to match model's state_dim
        for t in raw:
            if t['states'].shape[1] < sd:
                p = np.zeros((t['states'].shape[0], sd), np.float32)
                p[:, :t['states'].shape[1]] = t['states']
                t['states'] = p
        agent.replay_buffer = raw[-args.buffer_size:]
        print(f'Loaded {len(agent.replay_buffer)} expert trajectories into buffer')

    layout_names = ['mediumClassic', 'smallClassic', 'trappedClassic']
    layouts = [layout.getLayout(n) for n in layout_names]
    scores, wins = [], 0

    for ep in range(args.episodes):
        idx = ep % len(layouts)
        lo, ln = layouts[idx], layout_names[idx]
        t0 = time.time()
        traj = agent.play_episode(lo, epsilon=args.epsilon * (1 - ep / args.episodes))
        dt_ep = time.time() - t0

        if traj is None: continue
        # Ensure states are padded to state_dim
        if traj['states'].shape[1] < sd:
            p = np.zeros((traj['states'].shape[0], sd), np.float32)
            p[:, :traj['states'].shape[1]] = traj['states']
            traj['states'] = p
        agent.replay_buffer.append(traj)
        if len(agent.replay_buffer) > args.buffer_size:
            agent.replay_buffer.pop(0)

        scores.append(traj['score'])
        if traj['win']: wins += 1
        avg10 = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        wr = wins / len(scores) * 100
        print(f'[{ep+1:3d}/{args.episodes}] {ln:15s} '
              f'score={traj["score"]:7.0f}  avg10={avg10:7.0f}  '
              f'win={str(traj["win"]):5s}  wr={wr:5.1f}%  [{dt_ep:.0f}s]')

        # Train every N episodes
        if (ep + 1) % args.train_every == 0 and len(agent.replay_buffer) >= 10:
            # Sample from buffer
            buf = agent.replay_buffer
            batch_trajs = np.random.choice(buf, min(args.batch_size, len(buf)), replace=False)
            K = 20
            b_rtg = np.zeros((len(batch_trajs), K, 1), np.float32)
            b_states = np.zeros((len(batch_trajs), K, sd), np.float32)
            b_acts = np.zeros((len(batch_trajs), K, 5), np.float32)
            b_mask = np.ones((len(batch_trajs), K), np.float32)

            for i, bt in enumerate(batch_trajs):
                T = len(bt['actions'])
                start = np.random.randint(0, max(1, T - K + 1))
                end = min(start + K, T); length = end - start
                offset = K - length
                sl = slice(offset, K)
                src_sl = slice(start, end)
                b_rtg[i, sl, 0] = bt['returns_to_go'][src_sl]
                b_states[i, sl] = bt['states'][src_sl]
                for j, aid in enumerate(bt['actions'][src_sl]):
                    b_acts[i, offset + j, int(aid)] = 1.0
                b_mask[i, :offset] = 0.0

            loss = agent.train_step(
                torch.FloatTensor(b_rtg).to(device),
                torch.FloatTensor(b_states).to(device),
                torch.FloatTensor(b_acts).to(device),
                torch.zeros(len(batch_trajs), K, dtype=torch.long).to(device),
                torch.FloatTensor(b_mask).to(device),
                opt)
            print(f'          train loss={loss:.4f}  buffer={len(buf)}')

        # Save checkpoint
        if (ep + 1) % 50 == 0:
            ckpt_path = os.path.join(PROJECT, f'checkpoints/dt_online_{ep+1}.pt')
            torch.save({'model_state_dict': dt.state_dict(), 'state_dim': sd,
                        'rtg_min': ckpt['rtg_min'], 'rtg_max': ckpt['rtg_max']}, ckpt_path)
            print(f'          saved {ckpt_path}')

    # Final save
    final = os.path.join(PROJECT, 'checkpoints/dt_online_final.pt')
    torch.save({'model_state_dict': dt.state_dict(), 'state_dim': sd,
                'rtg_min': ckpt['rtg_min'], 'rtg_max': ckpt['rtg_max']}, final)
    print(f'\nDone. Avg score: {np.mean(scores):.0f}  Win rate: {wins/len(scores)*100:.1f}%')
    print(f'Saved: {final}')


if __name__ == '__main__':
    main()
