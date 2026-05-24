"""
Evaluation script — runs the trained PacmanAgent on multiple episodes.

Usage:
    cd e:\Pacman
    python scripts/evaluate.py --checkpoint checkpoints/dt_ppo_final.pt --episodes 100
"""

import sys, os, argparse, time
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT)
sys.path.insert(0, SKELETON)

import layout
from pacman import GameState
from game import Directions
import ghostAgents

from src.model.decision_transformer import DecisionTransformer
from src.model.world_model import WorldModel
from src.agent.pacman_agent import PacmanAgent

# Feature extractor (same as collect_expert_data / train_ppo)
def extract_features(gameState):
    walls = gameState.getWalls()
    H, W = walls.height, walls.width
    px, py = gameState.getPacmanPosition()
    pacman_feat = np.array([px / H, py / W], dtype=np.float32)
    ghost_states = gameState.getGhostStates()
    ghost_pos, ghost_scared = [], []
    for g in ghost_states:
        gx, gy = g.getPosition()
        ghost_pos.extend([gx / H, gy / W])
        ghost_scared.append(g.scaredTimer / 40.0)
    food_flat = np.array(gameState.getFood().data, dtype=np.float32).flatten()
    caps = gameState.getCapsules()
    capsule_grid = np.zeros((W, H), dtype=np.float32)
    for cx, cy in caps:
        capsule_grid[cx, cy] = 1.0
    return np.concatenate([
        pacman_feat, np.array(ghost_pos, dtype=np.float32),
        np.array(ghost_scared, dtype=np.float32),
        food_flat, capsule_grid.flatten(),
    ]).astype(np.float32)


def get_state_dim(layout_obj):
    H, W = layout_obj.height, layout_obj.width
    G = layout_obj.getNumGhosts()
    return 2 + 2 * G + G + H * W + H * W


def run_episode(agent, layout_obj, verbose=False):
    """Run one episode. Returns (score, steps, win)."""
    num_ghosts = layout_obj.getNumGhosts()
    state = GameState()
    state.initialize(layout_obj, num_ghosts)
    ghost_agents = [ghostAgents.DirectionalGhost(i + 1, 0.8, 0.8)
                    for i in range(num_ghosts)]

    agent.reset()
    step = 0

    while not (state.isWin() or state.isLose()):
        action_id, debug = agent.act(
            state, extract_features,
            legal_actions=list(range(5)))
        action_str = {
            0: Directions.NORTH, 1: Directions.SOUTH,
            2: Directions.EAST,  3: Directions.WEST,
            4: Directions.STOP,
        }.get(action_id, Directions.STOP)

        state = state.generateSuccessor(0, action_str)
        reward = state.getScore() - (agent._last_value * 0 + 0)  # placeholder
        # Real reward tracking: use score delta
        agent.observe(reward)

        if state.isWin() or state.isLose():
            break

        for gi, ghost in enumerate(ghost_agents):
            if state.isWin() or state.isLose():
                break
            ga = ghost.getAction(state)
            state = state.generateSuccessor(gi + 1, ga or Directions.STOP)

        step += 1
        if step > 2000:
            break

    return state.getScore(), step, state.isWin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/dt_bc_final.pt')
    parser.add_argument('--layout', default='mediumClassic')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--mcts_simulations', type=int, default=50)
    parser.add_argument('--confidence_threshold', type=float, default=0.8)
    parser.add_argument('--no_mcts', action='store_true', help='Disable MCTS')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Layout ----
    layout_obj = layout.getLayout(args.layout)
    state_dim = get_state_dim(layout_obj)
    print(f"Layout: {args.layout}, state_dim={state_dim}")

    # ---- Model ----
    model = DecisionTransformer(
        state_dim=state_dim, act_dim=5,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, context_len=args.context_len,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Loaded: {args.checkpoint}")
    else:
        print(f"WARNING: checkpoint not found: {args.checkpoint}")
        print("  Using untrained model (will perform randomly)")

    # ---- WorldModel (dummy if not trained) ----
    world_model = WorldModel(state_dim, 5) if not args.no_mcts else None

    # ---- Agent ----
    agent = PacmanAgent(
        model, world_model,
        context_len=args.context_len,
        state_dim=state_dim, act_dim=5,
        target_rtg=500.0,
        confidence_threshold=args.confidence_threshold,
        mcts_simulations=args.mcts_simulations,
        device=device,
    )

    # ---- Run ----
    scores, wins, steps_arr = [], 0, []
    t0 = time.time()

    for ep in range(args.episodes):
        score, steps, win = run_episode(agent, layout_obj, verbose=args.verbose)
        scores.append(score)
        steps_arr.append(steps)
        if win:
            wins += 1

        if (ep + 1) % 10 == 0 or args.verbose:
            recent = np.mean(scores[-10:])
            print(f"[{ep+1:4d}/{args.episodes}] score={score:7.1f}  "
                  f"avg10={recent:7.1f}  win_rate={wins/(ep+1):.2f}")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Results ({args.episodes} episodes, {elapsed:.1f}s)")
    print(f"  Avg score:  {np.mean(scores):.1f}")
    print(f"  Std score:  {np.std(scores):.1f}")
    print(f"  Win rate:   {wins}/{args.episodes} ({wins/args.episodes:.2f})")
    print(f"  Avg steps:  {np.mean(steps_arr):.0f}")
    print(f"  Best:       {np.max(scores):.1f}")
    print(f"  Worst:      {np.min(scores):.1f}")


if __name__ == '__main__':
    main()
