"""
Expert Iteration — Round 1:
DT_v1 guides MCTS to collect high-quality data.

Usage:
    python scripts/collect_mcts_dt.py --checkpoint checkpoints/dt_v1_100ep.pt --episodes 30
"""

import sys, os, time, argparse
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL); sys.path.insert(0, PROJECT)
os.chdir(SKEL)

import layout, ghostAgents
from game import Directions, Game
from pacman import GameState, ClassicGameRules
from src.model.decision_transformer import DecisionTransformer
from src.planning.mcts import MCTS, MCTSNode
from src.model.world_model import WorldModel
from scripts.collect_expert_data import extract_features, get_state_shape

ACTION_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
ACTION_REV = {v: k for k, v in ACTION_MAP.items()}


class DTPriorAgent:
    """Agent that uses MCTS guided by DT prior and WorldModel rollouts."""
    def __init__(self, dt, wm, state_dim, n_sim=150, device='cpu'):
        self.mcts = MCTS(wm, dt, state_dim=state_dim, act_dim=5,
                         n_simulations=n_sim, rollout_depth=15, discount=0.99,
                         device=device)
        self.state_dim = state_dim
        self.dt = dt
        self.wm = wm

    def getAction(self, gameState):
        feat = extract_features(gameState).astype(np.float32)
        if len(feat) != self.state_dim:
            p = np.zeros(self.state_dim, dtype=np.float32)
            p[:len(feat)] = feat; feat = p
        legal = [a for a in gameState.getLegalActions(0) if a != Directions.STOP
                 or len(gameState.getLegalActions(0)) == 1]
        legal_ids = [ACTION_MAP[a] for a in legal]
        action_id, _ = self.mcts.search(feat, legal_actions=legal_ids)
        return ACTION_REV[action_id]


def collect_episode(agent, lo, state_dim):
    """Manual game loop, returns trajectory or None."""
    ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(lo.getNumGhosts())]
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    states, acts, rews = [], [], []
    prev = state.getScore(); step = 0

    while not (state.isWin() or state.isLose()) and step < 800:
        action = agent.getAction(state)
        state = state.generateSuccessor(0, action)
        cur = state.getScore(); rews.append(cur - prev); prev = cur
        feat = extract_features(state).astype(np.float32)
        if len(feat) != state_dim:
            p = np.zeros(state_dim, np.float32); p[:len(feat)] = feat; feat = p
        states.append(feat); acts.append(ACTION_MAP[action])
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        step += 1
        if step % 100 == 0:
            print(f'          step {step}...')

    if len(states) < 5: return None
    T = len(states)
    s = np.array(states, np.float32); a = np.array(acts, np.int32); r = np.array(rews, np.float32)
    rtg = np.zeros(T, np.float32); run = 0.0
    for t in reversed(range(T)): run += r[t]; rtg[t] = run
    return {'states': s, 'actions': a, 'rewards': r, 'returns_to_go': rtg,
            'steps': T, 'score': state.getScore(), 'win': state.isWin(),
            'source': 'mcts_dt_v1', 'quality_weight': 1.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/dt_v1_100ep.pt')
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--n_sim', type=int, default=150)
    p.add_argument('--output', default='data/iter1_mcts.npz')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  MCTS sims: {args.n_sim}')

    # Load DT — resolve relative paths against PROJECT
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dim = ckpt['state_dim']
    dt = DecisionTransformer(state_dim=state_dim, act_dim=5, d_model=256, n_heads=4, n_layers=4, context_len=20).to(device)
    dt.load_state_dict(ckpt['model_state_dict'])
    dt.eval()
    wm = WorldModel(state_dim, 5).to(device)

    agent = DTPriorAgent(dt, wm, state_dim, n_sim=args.n_sim, device=device)

    layouts = ['mediumClassic', 'smallClassic', 'trappedClassic']
    trajs = []
    for ep in range(args.episodes):
        ln = layouts[ep % len(layouts)]
        lo = layout.getLayout(ln)
        t0 = time.time()
        print(f'  [{ep+1}/{args.episodes}] {ln}  MCTS(DT_v1) sims={args.n_sim}')
        traj = collect_episode(agent, lo, state_dim)
        dt_ep = time.time() - t0
        if traj:
            trajs.append(traj)
            print(f'           steps={traj["steps"]}  score={traj["score"]:.0f}  win={traj["win"]}  [{dt_ep:.0f}s]')
        else:
            print(f'           too short, skipped')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, trajectories=np.array(trajs, dtype=object))
    wins = sum(1 for t in trajs if t['win'])
    avg_score = np.mean([t['score'] for t in trajs])
    print(f'\nDone: {len(trajs)} episodes  |  Avg score: {avg_score:.0f}  |  Wins: {wins}/{len(trajs)}')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
