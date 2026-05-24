"""
Human game recorder — play Pacman and save trajectories for DT training.

Controls: Arrow keys to move, close the window to stop.
Each completed episode is saved as data/human_XXX.npz

Usage:
    python scripts/human_record.py                         # mediumClassic (2 ghosts)
    python scripts/human_record.py --layout originalClassic  # 28x27, 4 ghosts
"""

import sys, os, time, pickle, argparse
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_LAYOUTS = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'search', 'layouts')
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL)

import layout
from game import Directions, Game
from pacman import GameState, ClassicGameRules
from graphicsDisplay import PacmanGraphics
from keyboardAgents import KeyboardAgent
import ghostAgents

from scripts.collect_expert_data import extract_features, get_state_shape

ACT_MAP = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}


class RecordingKeyboardAgent(KeyboardAgent):
    """Keyboard agent that also records (state, action, reward) transitions."""

    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.prev_score = 0

    def getAction(self, state):
        action = super().getAction(state)
        # Record
        feat = extract_features(state).astype(np.float32)
        self.states.append(feat)
        self.actions.append(ACT_MAP.get(action, 4))
        self.rewards.append(0.0)  # filled in after step
        return action

    def record_reward(self, reward):
        if self.rewards:
            self.rewards[-1] = reward

    def save_trajectory(self, score, output_dir):
        """Save recorded trajectory to .npz."""
        if len(self.actions) < 5:
            return None

        T = len(self.actions)
        s = np.array(self.states[:T], dtype=np.float32)
        a = np.array(self.actions[:T], dtype=np.int32)
        r = np.array(self.rewards[:T], dtype=np.float32)

        rtg = np.zeros(T, dtype=np.float32)
        run = 0.0
        for t in reversed(range(T)):
            run += r[t]
            rtg[t] = run

        idx = len([f for f in os.listdir(output_dir) if f.startswith('human_')])
        path = os.path.join(output_dir, f'human_{idx+1:03d}.npz')
        traj = {'states': s, 'actions': a, 'rewards': r, 'returns_to_go': rtg,
                'steps': T, 'score': score, 'win': score > 0,
                'source': 'human', 'quality_weight': 3.0}

        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(path, trajectories=np.array([traj], dtype=object))
        print(f'\n  Saved: {path}  (steps={T}, score={score})\n')
        return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='mediumClassic')
    p.add_argument('--output', default='data')
    p.add_argument('--speed', type=float, default=0.05)
    args = p.parse_args()

    # Try loading layout from search/layouts if not found in multiagent
    lo = layout.getLayout(args.layout)
    if lo is None:
        search_path = os.path.join(SEARCH_LAYOUTS, args.layout + '.lay')
        if os.path.exists(search_path):
            lo = layout.Layout(open(search_path).read().split('\n'))
            print(f'Loaded layout from search module: {args.layout}')

    if lo is None:
        print(f'Layout not found: {args.layout}')
        return

    num_ghosts = lo.getNumGhosts()
    print(f'Layout: {args.layout}  {lo.width}x{lo.height}  ghosts={num_ghosts}')
    print(f'Arrow keys to move.  Close window to stop.')
    print(f'Output: {args.output}/human_*.npz')

    output_dir = os.path.join(PROJECT, args.output)

    while True:
        agent = RecordingKeyboardAgent()
        ghosts = [ghostAgents.DirectionalGhost(i+1, 0.8, 0.8) for i in range(num_ghosts)]

        rules = ClassicGameRules()
        display = PacmanGraphics(zoom=1.0, frameTime=args.speed)
        game = rules.newGame(lo, agent, ghosts, display, quiet=False)
        game.run()

        final_score = game.state.getScore()
        # Fix rewards: reconstruct from move history
        # Replay to get actual step rewards
        replay_state = GameState()
        replay_state.initialize(lo, num_ghosts)
        prev = replay_state.getScore()
        agent.rewards = []
        for agent_idx, action in game.moveHistory:
            replay_state = replay_state.generateSuccessor(agent_idx, action)
            if agent_idx == 0:  # Pacman
                cur = replay_state.getScore()
                agent.rewards.append(cur - prev)
                prev = cur

        agent.save_trajectory(final_score, output_dir)
        print(f'  Score: {final_score}  |  Win: {final_score > 0}')
        print(f'  Press Enter to play again, or Ctrl+C to quit.')
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print('\nDone.')
            break


if __name__ == '__main__':
    main()
