#!/usr/bin/env python
"""
Pacman Expert Data Collection — Terminal Dashboard
===================================================
Press Ctrl+C anytime to gracefully stop and save.

Usage:
    python scripts/run_collection.py
    python scripts/run_collection.py --episodes 500 --fast
"""

import sys, os, time, signal, argparse, glob
from datetime import datetime, timedelta

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

# ── Configuration ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Pacman Expert Data Collection')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--fast', action='store_true', help='Use fast experts only (no d4/MCTS)')
    p.add_argument('--gui', action='store_true', help='Show graphical Pacman window')
    p.add_argument('--speed', type=float, default=0.05, help='GUI frame delay in seconds (0=fastest)')
    p.add_argument('--resume', action='store_true', help='Load latest inc_*.npz and continue')
    p.add_argument('--output', default='data')
    return p.parse_args()

def build_pool(fast=False):
    """Return (expert_pool_config, layout_weights, ghost_profiles)."""
    if fast:
        experts = [
            ('alphabeta_d4',  'd4', 1.0, 3),
            ('alphabeta_d3',  'd3', 0.8, 4),
            ('minimax_d3',    'm3', 0.6, 1),
        ]
    else:
        experts = [
            ('alphabeta_d4',  'd4', 1.0, 5),   # heavy d4 — best quality
            ('alphabeta_d3',  'd3', 0.8, 4),
            ('minimax_d4',    'm4', 0.7, 2),   # minimax d4 — stronger
            ('expectimax_d3', 'e3', 0.5, 1),
            ('mcts_80',       'MC', 1.2, 2),
        ]

    layouts = {
        'smallClassic':   3,   # 2 ghosts, 20x7  — fast
        'mediumClassic':  3,   # 2 ghosts, 20x11 — fast
        'trappedClassic': 2,   # 2 ghosts, 8x5   — tiny
    }

    ghosts = ['aggressive', 'balanced', 'random', 'coward']
    return experts, layouts, ghosts


# ── Dashboard ──────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, total):
        self.total = total
        self.collected = 0
        self.scores = []
        self.wins = 0
        self.t0 = time.time()
        self.ep_times = []
        self.expert_counts = {}
        self.ghost_counts = {}

    def update(self, score, win, steps, expert, ghost, dt):
        self.collected += 1
        self.scores.append(score)
        self.wins += int(win)
        self.ep_times.append(dt)
        self.expert_counts[expert] = self.expert_counts.get(expert, 0) + 1
        self.ghost_counts[ghost] = self.ghost_counts.get(ghost, 0) + 1

    def render(self):
        n = self.collected
        if n == 0:
            return "No episodes collected yet."
        elapsed = time.time() - self.t0
        rate = n / (elapsed / 60) if elapsed > 0 else 0
        eta = (self.total - n) / rate * 60 if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # progress bar
        w = 40
        f = int(w * n / self.total)
        bar = '#' * f + '-' * (w - f)

        lines = []
        lines.append(f'  [{bar}]  {n}/{self.total}  ({n/self.total*100:.1f}%)')
        lines.append(f'  Elapsed: {elapsed_str}  |  Rate: {rate:.1f} ep/min  |  ETA: {eta_str}')
        lines.append(f'  Avg Score: {np.mean(self.scores):.0f}  |  Win Rate: {self.wins/n*100:.1f}%  |  Avg Time: {np.mean(self.ep_times):.0f}s/ep')
        lines.append(f'  Experts: {self._fmt_counts(self.expert_counts)}')
        lines.append(f'  Ghosts:  {self._fmt_counts(self.ghost_counts)}')
        return '\n'.join(lines)

    def _fmt_counts(self, d):
        return '  '.join(f'{k}:{v}' for k, v in sorted(d.items()))


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    experts, layouts, ghosts = build_pool(args.fast)
    total = args.episodes

    # Import collection engine — insert multiagent path so imports resolve
    SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
    sys.path.insert(0, SKEL)
    os.chdir(SKEL)

    import layout as lay
    import ghostAgents as ga_mod
    from multiAgents import MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent
    from scripts.collect_expert_data import (
        collect_one_episode, extract_features, _MCTSWrapper
    )

    # Build experts
    expert_pool = []
    for tag, short, qual, prob in experts:
        if tag.startswith('alphabeta'):
            depth = tag.split('_d')[1]
            expert_pool.append((tag, short, lambda d=depth: AlphaBetaAgent(depth=d, evalFn='betterEvaluationFunction'), qual, prob))
        elif tag.startswith('minimax'):
            depth = tag.split('_d')[1]
            expert_pool.append((tag, short, lambda d=depth: MinimaxAgent(depth=d, evalFn='betterEvaluationFunction'), qual, prob))
        elif tag.startswith('mcts'):
            expert_pool.append((tag, short, lambda: _MCTSWrapper(80), qual, prob))
        elif tag.startswith('expectimax'):
            depth = tag.split('_d')[1]
            expert_pool.append((tag, short, lambda d=depth: ExpectimaxAgent(depth=d), qual, prob))
        elif tag == 'mcts_80':
            expert_pool.append((tag, short, lambda: _MCTSWrapper(80), qual, prob))

    # Normalize expert probs
    probs = np.array([p for _, _, _, _, p in expert_pool], dtype=np.float32)
    probs /= probs.sum()

    # Build layout probs
    layout_names = list(layouts.keys())
    layout_w = np.array([layouts[n] for n in layout_names], dtype=np.float32)
    layout_w /= layout_w.sum()

    # Ghost factories
    def make_ghost_factory(profile):
        if profile == 'random':
            return lambda lo: [ga_mod.RandomGhost(i+1) for i in range(lo.getNumGhosts())]
        attack = {'aggressive': 0.9, 'balanced': 0.5, 'coward': 0.2}[profile]
        flee = {'aggressive': 0.2, 'balanced': 0.5, 'coward': 0.9}[profile]
        return lambda lo: [ga_mod.DirectionalGhost(i+1, attack, flee) for i in range(lo.getNumGhosts())]

    ghost_factories = {g: make_ghost_factory(g) for g in ghosts}

    # State
    output_dir = os.path.join(PROJECT, args.output)
    os.makedirs(output_dir, exist_ok=True)
    all_trajectories = []
    resume_count = 0

    # Resume from latest checkpoint
    if args.resume:
        inc_files = sorted(glob.glob(os.path.join(output_dir, 'inc_*.npz')))
        if inc_files:
            latest = inc_files[-1]
            data = np.load(latest, allow_pickle=True)
            all_trajectories = list(data['trajectories'])
            resume_count = len(all_trajectories)
            print(f'Resumed: loaded {resume_count} episodes from {os.path.basename(latest)}')
        else:
            print('No checkpoint found, starting fresh')

    instances = {}
    dash = Dashboard(total)
    dash.collected = resume_count
    # Pre-populate dashboard stats from loaded trajectories
    for t in all_trajectories:
        dash.scores.append(t['score'])
        dash.wins += int(t['win'])
        dash.ep_times.append(0)  # historical, no timing
        src = t.get('source', '?')
        dash.expert_counts[src] = dash.expert_counts.get(src, 0) + 1
        gho = t.get('ghost_profile', '?')
        dash.ghost_counts[gho] = dash.ghost_counts.get(gho, 0) + 1

    # Action mapping
    from game import Directions as Dirs
    ACTION_MAP2 = {Dirs.NORTH: 0, Dirs.SOUTH: 1, Dirs.EAST: 2, Dirs.WEST: 3, Dirs.STOP: 4}

    # GUI
    display = None
    if args.gui:
        import graphicsDisplay
        display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime=args.speed)

    # Graceful shutdown
    stopped = False
    def on_sig(sig, frame):
        nonlocal stopped
        stopped = True
        print('\n[Ctrl+C] Stopping gracefully...')
    signal.signal(signal.SIGINT, on_sig)

    # ── GUI episode runner ──
    def run_episode_gui(agent, ghost_fac, layout_obj):
        """Run one episode using the skeleton's Game class (supports graphics)."""
        from game import Game
        from pacman import ClassicGameRules, GameState

        num_g = layout_obj.getNumGhosts()
        ghosts = ghost_fac(layout_obj)
        rules = ClassicGameRules()
        game = rules.newGame(layout_obj, agent, ghosts, display or None, quiet=False)
        game.run()
        state = game.state
        T = len(game.moveHistory)
        if T < 5:
            return None
        # Reconstruct trajectory from move history
        s = GameState()
        s.initialize(layout_obj, num_g)
        states, acts, rews = [], [], []
        prev_score = s.getScore()
        for agent_idx, action in game.moveHistory:
            s = s.generateSuccessor(agent_idx, action)
            if agent_idx == 0:  # Pacman
                cur = s.getScore()
                rews.append(cur - prev_score)
                prev_score = cur
                states.append(extract_features(s))
                acts.append(ACTION_MAP2.get(action, 4))
        if len(states) < 5:
            return None
        s_arr = np.array(states, dtype=np.float32)
        a_arr = np.array(acts, dtype=np.int32)
        r_arr = np.array(rews, dtype=np.float32)
        rtg = np.zeros(len(r_arr), dtype=np.float32)
        run = 0.0
        for t in reversed(range(len(r_arr))):
            run += r_arr[t]; rtg[t] = run
        return {'states': s_arr, 'actions': a_arr, 'rewards': r_arr, 'returns_to_go': rtg,
                'steps': len(states), 'score': state.getScore(), 'win': state.isWin()}

    # ── Header ──
    print('=' * 70)
    print('  Pacman Expert Data Collection Dashboard')
    print(f'  Target: {total} episodes  |  Mode: {"Fast" if args.fast else "Full"}'
          + ('  |  GUI' if args.gui else ''))
    print(f'  Experts: {len(expert_pool)}  |  Layouts: {len(layout_names)}  |  Ghosts: {len(ghosts)}')
    print(f'  Output:  {output_dir}/')
    print('=' * 70)
    print()

    # ── Collection loop ──
    while dash.collected < total and not stopped:
        # Pick layout, expert, ghost
        layout_name = np.random.choice(layout_names, p=layout_w)
        layout_obj = lay.getLayout(layout_name)
        if layout_obj is None:
            continue

        idx = np.random.choice(len(expert_pool), p=probs)
        tag, short, factory, qual, _ = expert_pool[idx]
        if tag not in instances:
            instances[tag] = factory()
        agent = instances[tag]

        ghost_tag = np.random.choice(ghosts)
        ghost_fac = ghost_factories[ghost_tag]

        # Run episode (gui or headless)
        t0 = time.time()
        if display:
            traj = run_episode_gui(agent, ghost_fac, layout_obj)
        else:
            traj = collect_one_episode(agent, ghost_fac, layout_obj, min_steps=5)
        dt = time.time() - t0

        if traj is None:
            continue

        # Tag & weight
        traj['source'] = tag
        traj['ghost_profile'] = ghost_tag
        qw = qual
        if traj['win']: qw *= 2.0
        elif traj['steps'] >= 500: qw *= 1.3
        traj['quality_weight'] = qw

        all_trajectories.append(traj)
        dash.update(traj['score'], traj['win'], traj['steps'], short, ghost_tag[:3], dt)

        # ── Display ──
        os.system('cls' if os.name == 'nt' else 'clear')
        print('=' * 70)
        print('  Pacman Expert Data Collection Dashboard')
        print('=' * 70)
        print(dash.render())
        print(f'\n  Last: {layout_name} | {short} | ghost={ghost_tag} | '
              f'steps={traj["steps"]} score={traj["score"]:.0f} win={traj["win"]} [{dt:.0f}s]')
        if dash.collected % 10 == 0:
            inc = os.path.join(output_dir, f'inc_{dash.collected:04d}.npz')
            np.savez_compressed(inc, trajectories=np.array(all_trajectories, dtype=object))
            print(f'  Saved checkpoint: inc_{dash.collected:04d}.npz')
        print()

    # ── Final save ──
    final = os.path.join(output_dir, 'expert_trajectories.npz')
    np.savez_compressed(final, trajectories=np.array(all_trajectories, dtype=object))
    print(f'Done. Saved {dash.collected} episodes to {final}')


if __name__ == '__main__':
    import numpy as np
    main()
