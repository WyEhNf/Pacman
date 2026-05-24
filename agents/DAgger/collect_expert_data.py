"""
Expert Data Collection Script
==============================
Runs a diverse pool of expert agents with varied ghost behaviour on
multiple Pacman layouts and saves tagged trajectories for Decision
Transformer Phase-1 (Behavioral Cloning) training.

Key design decisions:
 - Multiple expert types (AlphaBeta, Minimax, Expectimax, A*) sampled
   per episode → DT sees diverse decision styles.
 - Multiple ghost profiles (aggressive, balanced, random, coward) sampled
   per episode → DT generalises to unseen opponents.
 - Each trajectory tagged with `source`, `ghost_profile` and
   `quality_weight` so the training loop can up-weight high-quality data.

Usage:
    cd e:\Pacman
    python scripts/collect_expert_data.py --episodes 500
"""

import sys
import os
import random, time
import numpy as np
import argparse
from collections import Counter

# ─── Path Setup ─────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIAGENT = os.path.join(PROJECT_ROOT, 'PPCA-AIPacMan-2024-main', 'multiagent')
SEARCH     = os.path.join(PROJECT_ROOT, 'PPCA-AIPacMan-2024-main', 'search')
sys.path.insert(0, SEARCH)
sys.path.insert(0, MULTIAGENT)  # MULTIAGENT must be first (game engine)

# layout.getLayout() resolves paths relative to CWD, so switch to the
# multiagent directory first.
_original_cwd = os.getcwd()
os.chdir(MULTIAGENT)

import layout
from pacman import GameState
from game import Directions
import ghostAgents as ghost_module


# ─── Action Encoding ────────────────────────────────────────────────────

ACTION_MAP = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST:  2,
    Directions.WEST:  3,
    Directions.STOP:  4,
}


# ═══════════════════════════════════════════════════════════════════════
#  Expert Pool  —  diverse Pacman decision-makers
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (source_tag, factory_fn, quality_weight, sampling_prob)
#
# quality_weight:    how much DT should trust this trajectory (higher = more)
# sampling_prob:     how often this expert is picked per episode.
#                    Probabilities are normalised internally, so feel free
#                    to use relative weights (1/2/3/4).
#
# Philosophy:
#  - High-depth AlphaBeta is the "gold standard"; it appears most often
#    and carries the highest weight.
#  - Shallower searches and Expectimax add behavioural diversity: the DT
#    sees what "good enough" looks like and learns to adapt its play-style.
#  - A* (pure navigation) provides path-to-food patterns without ghost
#    awareness.  Combined with ghost-aware experts, the DT learns to
#    interpolate between "eat efficiently" and "play safe".

EXPERT_POOL = []  # built below to keep factory clean


def _make_experts():
    """Populate EXPERT_POOL.  Called once at module load."""
    from multiAgents import MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent

    pool = [
        # (tag,                    factory,                           quality, prob)
        ('alphabeta_d4',  lambda: AlphaBetaAgent(depth='4', evalFn='betterEvaluationFunction'),          1.0,     2),
        ('alphabeta_d3',  lambda: AlphaBetaAgent(depth='3', evalFn='betterEvaluationFunction'),          0.8,     5),
        ('alphabeta_d2',  lambda: AlphaBetaAgent(depth='2', evalFn='betterEvaluationFunction'),          0.4,     1),
        ('minimax_d3',    lambda: MinimaxAgent(depth='3', evalFn='betterEvaluationFunction'),            0.6,     2),
        ('expectimax_d3', lambda: ExpectimaxAgent(depth='3'),         0.5,     2),
    ]
    return pool


# A* food-search agent — lives in the search module, so we import lazily.
class _AStarFoodWrapper:
    """Thin wrapper so the search module's A* agent matches the getAction interface."""
    def __init__(self):
        from searchAgents import AStarFoodSearchAgent
        self._agent = AStarFoodSearchAgent()
        self._registered = False

    def getAction(self, state):
        if not self._registered:
            self._agent.registerInitialState(state)
            self._registered = True
        return self._agent.getAction(state)


# Simple MCTS agent for high-quality expert data (no DT needed)
class _MCTSWrapper:
    """MCTS with random rollout.  Slower but produces strong play."""
    def __init__(self, n_simulations=80, rollout_depth=30):
        from game import Directions as Dirs
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.Dirs = Dirs

    def getAction(self, gameState):
        from game import Directions as Dirs
        import random as rnd
        root = gameState
        legal = [a for a in root.getLegalActions(0) if a != Dirs.STOP or len(root.getLegalActions(0)) == 1]
        if not legal:
            return Dirs.STOP

        best_action = legal[0]
        best_score = -float('inf')
        for action in legal:
            total = 0.0
            for _ in range(self.n_simulations):
                s = root.generateSuccessor(0, action)
                for _ in range(self.rollout_depth):
                    if s.isWin() or s.isLose():
                        break
                    # Ghosts move
                    for gi in range(1, s.getNumAgents()):
                        gl = s.getLegalActions(gi)
                        if gl:
                            s = s.generateSuccessor(gi, rnd.choice(gl))
                        if s.isWin() or s.isLose():
                            break
                    if s.isWin() or s.isLose():
                        break
                    # Pacman random
                    pl = s.getLegalActions(0)
                    if pl:
                        s = s.generateSuccessor(0, rnd.choice(pl))
                total += s.getScore()
            avg = total / self.n_simulations
            if avg > best_score:
                best_score = avg
                best_action = action
        return best_action

EXPERT_POOL = _make_experts()
EXPERT_POOL.append(
    ('mcts_80',        lambda: _MCTSWrapper(80),                          1.2,     2),
)


# ═══════════════════════════════════════════════════════════════════════
#  Ghost Profiles  —  diverse opponent behaviour
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (profile_tag, factory_fn, description)

def _make_ghost_profile(tag, prob_attack, prob_scared_flee):
    """Return a factory that creates ghost agents for the given profile."""
    def factory(layout_obj):
        num = layout_obj.getNumGhosts()
        return [ghost_module.DirectionalGhost(i + 1, prob_attack, prob_scared_flee)
                for i in range(num)]
    return factory


GHOST_PROFILES = {
    'aggressive': {
        'factory': _make_ghost_profile('aggressive', prob_attack=0.9, prob_scared_flee=0.2),
        'desc': 'Ghosts chase Pacman aggressively, barely flee when scared',
    },
    'balanced': {
        'factory': _make_ghost_profile('balanced', prob_attack=0.5, prob_scared_flee=0.5),
        'desc': 'Mixed strategy — sometimes chase, sometimes wander',
    },
    'random': {
        'factory': lambda layout: [ghost_module.RandomGhost(i + 1)
                                    for i in range(layout.getNumGhosts())],
        'desc': 'Ghosts move completely at random',
    },
    'coward': {
        'factory': _make_ghost_profile('coward', prob_attack=0.2, prob_scared_flee=0.9),
        'desc': 'Ghosts avoid Pacman, flee quickly when scared',
    },
}


# ─── State Feature Extractor ────────────────────────────────────────────

def get_state_shape(layout_obj):
    H, W = layout_obj.height, layout_obj.width
    G = layout_obj.getNumGhosts()
    return 2 + 2 * G + G + H * W + H * W


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
    ghost_pos = np.array(ghost_pos, dtype=np.float32)
    ghost_scared = np.array(ghost_scared, dtype=np.float32)

    food_flat = np.array(gameState.getFood().data, dtype=np.float32).flatten()

    caps = gameState.getCapsules()
    # Capsule coords (x, y) = (column, row); grid uses [col][row]
    capsule_grid = np.zeros((W, H), dtype=np.float32)
    for cx, cy in caps:
        capsule_grid[cx][cy] = 1.0

    feats = np.concatenate([
        pacman_feat, ghost_pos, ghost_scared, food_flat, capsule_grid.flatten(),
    ])
    return feats.astype(np.float32)


# ─── Single Episode ─────────────────────────────────────────────────────

def collect_one_episode(pacman_agent, ghost_factory, layout_obj,
                        min_steps=5, max_steps=800):
    """Run one episode.  Returns trajectory dict or None."""
    num_ghosts = layout_obj.getNumGhosts()
    state = GameState()
    state.initialize(layout_obj, num_ghosts)
    ghost_agents = ghost_factory(layout_obj)

    states, actions, rewards = [], [], []
    prev_score = state.getScore()
    step = 0

    while not (state.isWin() or state.isLose()):
        # Pacman
        action = pacman_agent.getAction(state)
        if action is None:
            action = Directions.STOP
        state = state.generateSuccessor(0, action)

        curr_score = state.getScore()
        rewards.append(curr_score - prev_score)
        prev_score = curr_score
        states.append(extract_features(state))
        actions.append(ACTION_MAP.get(action, 4))

        if state.isWin() or state.isLose():
            break

        # Ghosts
        for gi, ghost in enumerate(ghost_agents):
            if state.isWin() or state.isLose():
                break
            ga = ghost.getAction(state)
            if ga is None:
                ga = Directions.STOP
            state = state.generateSuccessor(gi + 1, ga)

        step += 1
        if step % 200 == 0 and step > 0:
            print(f'          ... step {step}/{max_steps} ...')
        if step >= max_steps:
            break

    if len(states) < min_steps:
        return None

    T = len(states)
    s = np.array(states, dtype=np.float32)
    a = np.array(actions, dtype=np.int32)
    r = np.array(rewards, dtype=np.float32)

    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running += r[t]
        rtg[t] = running

    return {
        'states':         s,
        'actions':        a,
        'rewards':        r,
        'returns_to_go':  rtg,
        'steps':          T,
        'score':          state.getScore(),
        'win':            state.isWin(),
    }


# ─── Batch Collection ───────────────────────────────────────────────────

DEFAULT_LAYOUTS = [
    'smallClassic', 'mediumClassic', 'trappedClassic',
]

LAYOUT_WEIGHTS = {
    'smallClassic':   3,
    'mediumClassic':  3,
    'trappedClassic': 2,
}


def collect_dataset(
    num_episodes=500,
    layout_names=None,
    min_steps=10,
    output_path=None,
    verbose=True,
):
    if layout_names is None:
        layout_names = DEFAULT_LAYOUTS
    # Build layout sampling probabilities
    raw_w = np.array([LAYOUT_WEIGHTS.get(name, 1) for name in layout_names], dtype=np.float32)
    raw_w[raw_w == 0] = 0  # keep zeros
    layout_probs = raw_w / raw_w.sum()
    # Filter out zero-weight layouts
    active = [(n, p) for n, p in zip(layout_names, layout_probs) if p > 0]
    layout_names = [n for n, _ in active]
    layout_probs = np.array([p for _, p in active], dtype=np.float32)
    layout_probs /= layout_probs.sum()

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, 'data', 'expert_trajectories.npz')

    _start_time = time.time()

    if verbose:
        print("=" * 60)
        print("Expert Data Collection  (diverse experts x diverse ghosts)")
        print(f"  Target:   {num_episodes} episodes")
        print(f"  Experts:  {[tag for tag, _, w, p in EXPERT_POOL]}")
        print(f"  Ghosts:   {list(GHOST_PROFILES.keys())}")
        print(f"  Layouts:  {', '.join(layout_names)}")
        print("=" * 60)

    # ── Validate first layout exists ──
    sample_layout = layout.getLayout(layout_names[0])
    if sample_layout is None:
        raise RuntimeError(
            f"Cannot find layout '{layout_names[0]}'. "
            f"Check PPCA-AIPacMan-2024-main/multiagent/layouts/")

    # ── Pre-build probability tables ──
    expert_tags    = [t for t, _, _, _ in EXPERT_POOL]
    expert_factories = [f for _, f, _, _ in EXPERT_POOL]
    expert_weights = [w for _, _, w, _ in EXPERT_POOL]
    expert_probs   = np.array([p for _, _, _, p in EXPERT_POOL], dtype=np.float32)
    expert_probs   /= expert_probs.sum()

    ghost_names     = list(GHOST_PROFILES.keys())
    ghost_factories = [GHOST_PROFILES[k]['factory'] for k in ghost_names]

    # ── Generate expert instances on first use (cached) ──
    expert_instances = {}  # expert_tag → agent instance

    all_trajectories = []
    expert_counts = Counter()
    ghost_counts  = Counter()
    scores = []
    wins = 0

    while len(all_trajectories) < num_episodes:
        # Random sample: layout → expert → ghost profile
        layout_name = np.random.choice(layout_names, p=layout_probs)

        layout_obj = layout.getLayout(layout_name)
        if layout_obj is None:
            continue

        # Pick expert
        chosen_expert_idx = np.random.choice(len(EXPERT_POOL), p=expert_probs)
        expert_tag = expert_tags[chosen_expert_idx]

        if expert_tag not in expert_instances:
            expert_instances[expert_tag] = expert_factories[chosen_expert_idx]()
        pacman_agent = expert_instances[expert_tag]

        # Pick ghost profile (uniform)
        ghost_tag = random.choice(ghost_names)
        ghost_factory = GHOST_PROFILES[ghost_tag]['factory']

        # Log start
        n = len(all_trajectories)
        t0_ep = time.time()
        tstr = time.strftime('%H:%M:%S')
        if verbose:
            print(f"  [{n+1}/{num_episodes}] {tstr}  START  {layout_name}  {expert_tag}  ghost={ghost_tag}")

        # Collect
        traj = collect_one_episode(pacman_agent, ghost_factory, layout_obj,
                                   min_steps=min_steps)
        dt_ep = time.time() - t0_ep

        if traj is None:
            continue

        # Tag trajectory with metadata
        traj['source']         = expert_tag
        traj['ghost_profile']  = ghost_tag
        qw = expert_weights[chosen_expert_idx]
        if traj['win']:
            qw *= 2.0  # double weight for winning trajectories
        elif traj['steps'] >= 500:
            qw *= 1.3  # bonus for long-survival episodes
        traj['quality_weight'] = qw

        all_trajectories.append(traj)
        expert_counts[expert_tag] += 1
        ghost_counts[ghost_tag]   += 1
        scores.append(traj['score'])
        if traj['win']:
            wins += 1

        n = len(all_trajectories)

        # ── Incremental save every 10 episodes ──
        if n % 10 == 0:
            inc_path = os.path.join(os.path.dirname(output_path),
                                    f'inc_{n:04d}.npz')
            np.savez_compressed(
                inc_path,
                trajectories=np.array(all_trajectories, dtype=object),
            )

        # ── Per-episode progress with timing ──
        if verbose:
            elapsed = time.time() - _start_time
            eps_per_min = n / (elapsed / 60.0) if elapsed > 0 else 0
            eta_min = (num_episodes - n) / eps_per_min if eps_per_min > 0 else 0

            # Progress bar
            bar_width = 30
            filled = int(bar_width * n / num_episodes)
            bar = '#' * filled + '-' * (bar_width - filled)

            # Format time
            def _fmt(m):
                if m < 60: return f'{m:.0f}m'
                return f'{m/60:.1f}h'
            eta_str = _fmt(eta_min)
            elapsed_str = _fmt(elapsed / 60.0)

            pct = n / num_episodes * 100
            status = (f'[{bar}] {pct:5.1f}%  [{n}/{num_episodes}]  '
                      f'{elapsed_str} elapsed  ~{eps_per_min:.0f}ep/m  '
                      f'ETA {eta_str}')

            ep_line = (f'  {layout_name:17s} {expert_tag:15s} '
                       f'ghost={ghost_tag:10s} '
                       f'steps={traj["steps"]:4d}  score={traj["score"]:7.1f}  '
                       f'win={traj["win"]}  [{dt_ep:.0f}s]')

            print(status)
            print(ep_line)
            if n % 5 == 0 and n > 0:
                avg = np.mean(scores[-min(20, n):])
                wr = wins / n * 100
                print(f'  ── recent20_avg={avg:.0f}  win_rate={wr:.1f}%')
            print()

    # ── Summary ──
    if verbose:
        print("-" * 60)
        print("Collection complete.")
        print(f"  Episodes:       {len(all_trajectories)}")
        print(f"  Avg score:      {np.mean(scores):.1f}")
        print(f"  Avg steps:      {np.mean([t['steps'] for t in all_trajectories]):.0f}")
        print(f"  Win rate:       {wins / len(all_trajectories) * 100:.1f}%")
        print()
        print("  Expert distribution:")
        for tag in expert_tags:
            n_e = expert_counts[tag]
            if n_e > 0:
                avg_s = np.mean([t['score'] for t in all_trajectories if t['source'] == tag])
                print(f"    {tag:18s}  {n_e:4d} ep  avg_score={avg_s:7.1f}")
        print()
        print("  Ghost profile distribution:")
        for tag in ghost_names:
            n_g = ghost_counts[tag]
            if n_g > 0:
                print(f"    {tag:18s}  {n_g:4d} ep")

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        trajectories=np.array(all_trajectories, dtype=object),
    )

    if verbose:
        mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  Saved to: {output_path}  ({mb:.1f} MB)")

    return all_trajectories


# ─── Main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect diverse expert trajectories for Decision Transformer training')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--layouts', type=str, nargs='+', default=None)
    parser.add_argument('--min-steps', type=int, default=10)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.episodes,
        layout_names=args.layouts,
        min_steps=args.min_steps,
        output_path=args.output,
        verbose=not args.quiet,
    )
