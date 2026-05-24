"""Phase A: DAgger_R1 ensemble self-play with full reward shaping decomposition.

Collects 100 episodes on mediumClassic with varied ghost profiles.
For every step, records all 8 reward components for analysis.

Usage:
    python scripts/phaseA_collect.py --episodes 100
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn, random, time
from collections import defaultdict

PROJECT = r'E:\Pacman'
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState

ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}
REV = {v: k for k, v in ACT.items()}
DIR_VEC = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
           Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}

H, W, C = 11, 20, 8

# ── Model ──
class CNNDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5))
    def forward(self, x): return self.fc(self.conv(x).mean(dim=[2, 3]))

# ── Precompute walls ──
lo = layout.getLayout('mediumClassic')

def get_walls_grid():
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg

WALLS = get_walls_grid()

def state_to_grid(state):
    g = np.zeros((C, H, W), dtype=np.float32)
    fd = state.getFood()
    for x in range(W):
        for y in range(H):
            if x < fd.width and y < fd.height and fd[x][y]: g[0, y, x] = 1.0
    for cx, cy in state.getCapsules():
        if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
    px, py = state.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0

    ghosts = state.getGhostStates()
    ranked = sorted(ghosts, key=lambda gh: abs(px - int(gh.getPosition()[0])) + abs(py - int(gh.getPosition()[1])))
    for i, gh in enumerate(ranked[:2]):
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < W and 0 <= gy < H:
            g[3 + i, gy, gx] = 1.0; g[5 + i, gy, gx] = gh.scaredTimer / 40.0
    g[7] = WALLS; return g

def load_models():
    models = []
    for i in range(5):
        m = CNNDQN()
        m.load_state_dict(torch.load(os.path.join(PROJECT, f'checkpoints/dagger_cnn_m{i}_final.pt'), map_location='cpu'))
        m.eval(); models.append(m)
    return models

# ── Ghost factories ──
def make_ghosts(profile):
    profiles = {
        'balanced':    (0.5, 0.5),
        'aggressive':  (0.9, 0.2),
        'coward':      (0.2, 0.9),
        'random':      None,
    }
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    a, f = profiles[profile]
    return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(lo.getNumGhosts())]

# ── Reward shaping ──
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def nearest_food_dist(state, px, py):
    fd = state.getFood()
    best = 999
    for x in range(fd.width):
        for y in range(fd.height):
            if fd[x][y]:
                d = abs(px - x) + abs(py - y)
                if d < best: best = d
    return best if best < 999 else 0

def min_ghost_dist(state, px, py):
    best = 999
    for g in state.getGhostStates():
        gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
        d = abs(px - gx) + abs(py - gy)
        if d < best: best = d
    return best

def nearest_scared_ghost_dist(state, px, py):
    best = 999
    for g in state.getGhostStates():
        if g.scaredTimer > 0:
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            d = abs(px - gx) + abs(py - gy)
            if d < best: best = d
    return best

def nearest_capsule_dist(state, px, py):
    caps = state.getCapsules()
    if not caps: return 999
    best = 999
    for cx, cy in caps:
        d = abs(px - cx) + abs(py - cy)
        if d < best: best = d
    return best

def any_nonscared_ghost_near(state, px, py, threshold=6):
    for g in state.getGhostStates():
        if g.scaredTimer <= 0:
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            if abs(px - gx) + abs(py - gy) <= threshold:
                return True
    return False

def compute_shaping(state, prev_state, action, prev_dir, prev_food_dist, prev_capsule_dist,
                    prev_scared_dist, prev_ghost_dist, prev_score):
    """Compute all reward shaping components for one step.
    Returns dict of {name: value} plus the new prev_* values.
    """
    px, py = state.getPacmanPosition()
    ghost_dist = min_ghost_dist(state, px, py)

    # ── Base: Δ game score ──
    curr_score = state.getScore()
    R_base = curr_score - prev_score

    # ── R_危险回避 (dense death prevention) ──
    if ghost_dist <= 2:
        R_danger = -3.0
    elif ghost_dist <= 4:
        R_danger = -1.0
    elif ghost_dist <= 6:
        R_danger = -0.3
    else:
        R_danger = 0.0

    # If game over (death), apply the actual death penalty (-500) — but also
    # the shaping has already been warning for the preceding steps.
    if state.isLose():
        R_death = -500.0
    else:
        R_death = 0.0

    # ── R_豆子吸引 (food navigation) ──
    food_dist = nearest_food_dist(state, px, py)
    R_food_nav = 0.0
    if prev_food_dist < 999 and food_dist < 999:
        R_food_nav = np.clip(0.3 * (prev_food_dist - food_dist), -3.0, 3.0)  # + if closer
    # Extra for actually eating a bean (detected via score delta from food)
    # Food = +10 in game score, but could also be capsule (+50) or ghost (+200)
    # Use food count change as a more reliable indicator
    prev_food_count = prev_state.getFood().count() if prev_state else food_dist
    curr_food_count = state.getFood().count()
    food_eaten = prev_food_count - curr_food_count
    R_food_eaten = 2.0 * food_eaten if food_eaten > 0 else 0.0
    # Also check ghost kills (scared ghost eaten)
    prev_ghost_count = len(prev_state.getGhostStates()) if prev_state else 2
    curr_ghost_count = len(state.getGhostStates())

    # ── R_胶囊引导 (ghost near + capsule exists → move toward capsule) ──
    capsule_dist = nearest_capsule_dist(state, px, py)
    R_capsule_guide = 0.0
    if any_nonscared_ghost_near(state, px, py, 6) and capsule_dist < 999:
        if prev_capsule_dist < 999 and capsule_dist < 999:
            R_capsule_guide = 1.0 * (prev_capsule_dist - capsule_dist)

    # ── R_追杀幽灵 (chase scared ghosts) ──
    scared_dist = nearest_scared_ghost_dist(state, px, py)
    R_chase = 0.0
    if scared_dist < 999:  # there is a scared ghost
        if prev_scared_dist < 999 and scared_dist < 999:
            R_chase = np.clip(1.5 * (prev_scared_dist - scared_dist), -3.0, 3.0)  # + if closer
        elif prev_scared_dist >= 999:  # just became scared
            R_chase = 1.5

    # ── R_动量 (momentum) ──
    R_momentum = 0.1 if (prev_dir and action == prev_dir) else 0.0

    # ── R_胜利 (win) ──
    R_win = 200.0 if state.isWin() else 0.0

    # ── R_时间 (step penalty) ──
    R_time = -0.05

    # ── Total ──
    R_total = (R_base + R_danger + R_death + R_food_nav + R_food_eaten +
               R_capsule_guide + R_chase + R_momentum + R_win + R_time)

    components = {
        'R_base':       R_base,
        'R_danger':     R_danger,
        'R_death':      R_death,
        'R_food_nav':   R_food_nav,
        'R_food_eaten': R_food_eaten,
        'R_capsule':    R_capsule_guide,
        'R_chase':      R_chase,
        'R_momentum':   R_momentum,
        'R_win':        R_win,
        'R_time':       R_time,
    }

    new_prev = {
        'food_dist': food_dist,
        'capsule_dist': capsule_dist,
        'scared_dist': scared_dist,
        'ghost_dist': ghost_dist,
        'score': curr_score,
    }
    return R_total, components, new_prev

# ── Episode runner ──
def run_episode(models, ghost_profile, seed, max_steps=500):
    """Run one episode collecting full reward decomposition."""
    random.seed(seed); np.random.seed(seed)
    ghosts = make_ghosts(ghost_profile)
    state = GameState(); state.initialize(lo, lo.getNumGhosts())

    grids, actions, rewards, r_components = [], [], [], []
    prev_dir = None
    step = 0

    # Initial state distances
    px, py = state.getPacmanPosition()
    prev = {
        'food_dist': nearest_food_dist(state, px, py),
        'capsule_dist': nearest_capsule_dist(state, px, py),
        'scared_dist': nearest_scared_ghost_dist(state, px, py),
        'ghost_dist': min_ghost_dist(state, px, py),
        'score': state.getScore(),
    }

    while not (state.isWin() or state.isLose()) and step < max_steps:
        # Record grid before action
        grids.append(state_to_grid(state))

        # Ensemble Q → pick action
        t = torch.FloatTensor(grids[-1]).unsqueeze(0)
        q = sum(m(t)[0].detach().numpy() for m in models) / len(models)
        legal = state.getLegalActions(0)
        ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
        if not ids: ids = [4]
        best, mv = -1e9, 4
        for i in range(5):
            if i in ids and q[i] > best: best = q[i]; mv = i
        action_dir = REV[mv]
        actions.append(mv)

        # Step pacman
        prev_state = state
        state = state.generateSuccessor(0, action_dir)

        # Compute rewards
        R_total, comps, prev = compute_shaping(
            state, prev_state, action_dir, prev_dir,
            prev['food_dist'], prev['capsule_dist'],
            prev['scared_dist'], prev['ghost_dist'], prev['score'])
        rewards.append(R_total)
        r_components.append(comps)
        prev_dir = action_dir

        if state.isWin() or state.isLose():
            break

        # Step ghosts
        for gi, gs in enumerate(ghosts):
            if state.isWin() or state.isLose():
                break
            state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)

        step += 1

    T = len(grids)
    if T < 5: return None

    return {
        'states': np.array(grids, dtype=np.float32),
        'actions': np.array(actions, dtype=np.int32),
        'rewards': np.array(rewards, dtype=np.float32),
        'r_components': r_components,
        'steps': T,
        'score': state.getScore(),
        'win': state.isWin(),
        'ghost_profile': ghost_profile,
    }

# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    N = args.episodes

    print('Loading DAgger R1 ensemble...')
    models = load_models()
    print(f'{len(models)} models loaded.\n')

    ghost_profiles = ['balanced', 'balanced', 'aggressive', 'coward', 'random']  # weighted toward balanced

    all_trajs = []
    scores, wins = [], 0

    # Accumulators for reward component stats
    comp_sums = defaultdict(float)
    comp_maxes = defaultdict(float)
    comp_mins = defaultdict(float)
    comp_zeros = defaultdict(int)
    total_steps = 0

    t0 = time.time()
    for ep in range(N):
        profile = ghost_profiles[ep % len(ghost_profiles)]
        traj = run_episode(models, profile, seed=ep)

        if traj is None: continue
        all_trajs.append(traj)
        scores.append(traj['score'])
        if traj['win']: wins += 1
        total_steps += traj['steps']

        # Accumulate component stats
        for comps in traj['r_components']:
            for k, v in comps.items():
                comp_sums[k] += v
                comp_maxes[k] = max(comp_maxes[k], v)
                comp_mins[k] = min(comp_mins[k], v)
                if v == 0: comp_zeros[k] += 1

        # Also track per-step total reward
        total_rewards_abs = [abs(r) for r in traj['rewards']]

        if (ep + 1) % 20 == 0:
            r20 = np.mean(scores[-20:])
            print(f'[{ep+1:3d}/{N}] last20_avg_score={r20:7.1f}  win_rate={wins/(ep+1):.2f}  '
                  f'avg_r_per_step={np.mean(total_rewards_abs):.2f}')

    elapsed = time.time() - t0

    # ── Save ──
    output_path = os.path.join(PROJECT, 'data', 'phaseA_reward_shaping.npz')
    np.savez_compressed(output_path, trajectories=np.array(all_trajs, dtype=object))
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nSaved: {output_path} ({mb:.1f} MB)')

    # ── Summary ──
    print(f'\n{"="*60}')
    print(f'PHASE A SUMMARY — {len(all_trajs)} episodes, {total_steps} total steps')
    print(f'{"="*60}')
    print(f'  Avg score:     {np.mean(scores):.1f}')
    print(f'  Std score:     {np.std(scores):.1f}')
    print(f'  Win rate:      {wins}/{len(all_trajs)} ({wins/len(all_trajs)*100:.1f}%)')
    print(f'  Avg steps/ep:  {np.mean([t["steps"] for t in all_trajs]):.0f}')
    print(f'  Time:          {elapsed:.0f}s')

    print(f'\n{"="*60}')
    print(f'REWARD COMPONENT STATISTICS (per step)')
    print(f'{"="*60}')
    print(f'{"Component":<18} {"Mean":>10} {"Min":>10} {"Max":>10} {"Zero%":>8}')
    print(f'{"-"*56}')
    n_steps = total_steps
    for name in ['R_base', 'R_danger', 'R_death', 'R_food_nav', 'R_food_eaten',
                 'R_capsule', 'R_chase', 'R_momentum', 'R_win', 'R_time']:
        mean_v = comp_sums[name] / n_steps if n_steps > 0 else 0
        zero_pct = comp_zeros[name] / n_steps * 100 if n_steps > 0 else 0
        print(f'{name:<18} {mean_v:10.4f} {comp_mins[name]:10.2f} {comp_maxes[name]:10.2f} {zero_pct:7.1f}%')

    # ── Total reward (game score only vs shaping only) ──
    all_r = []
    all_r_game = []   # R_base + R_death (native game score)
    all_r_shape = []  # everything else (our shaping)
    for t in all_trajs:
        for comps in t['r_components']:
            all_r.append(sum(comps.values()))
            all_r_game.append(comps['R_base'] + comps['R_death'])
            all_r_shape.append(sum(v for k, v in comps.items() if k not in ('R_base', 'R_death')))
    all_r = np.array(all_r)
    all_r_game = np.array(all_r_game)
    all_r_shape = np.array(all_r_shape)

    print(f'\n{"="*60}')
    print(f'TOTAL REWARD PER STEP')
    print(f'{"="*60}')
    print(f'{"":>20} {"Game Score (R_base+R_death)":>22} {"Shaping Only":>22}')
    print(f'{"  Mean":<20} {all_r_game.mean():22.4f} {all_r_shape.mean():22.4f}')
    print(f'{"  Std":<20} {all_r_game.std():22.4f} {all_r_shape.std():22.4f}')
    print(f'{"  Min":<20} {all_r_game.min():22.2f} {all_r_shape.min():22.2f}')
    print(f'{"  Max":<20} {all_r_game.max():22.2f} {all_r_shape.max():22.2f}')
    print(f'{"  P5":<20} {np.percentile(all_r_game, 5):22.4f} {np.percentile(all_r_shape, 5):22.4f}')
    print(f'{"  P50":<20} {np.percentile(all_r_game, 50):22.4f} {np.percentile(all_r_shape, 50):22.4f}')
    print(f'{"  P95":<20} {np.percentile(all_r_game, 95):22.4f} {np.percentile(all_r_shape, 95):22.4f}')

    shape_abs = np.abs(all_r_shape)
    print(f'\n  Shaping |R| > 3:  {(shape_abs > 3).mean()*100:.1f}%')
    print(f'  Shaping |R| > 5:  {(shape_abs > 5).mean()*100:.1f}%')

    print(f'\n  VERDICT: ', end='')
    if (shape_abs > 5).mean() < 0.01:
        print('PASS — shaping-only component |R| > 5 is < 1% of steps.')
        print(f'  Game score spikes (+10 food, +200 ghost, -500 death) are natural.')
        print(f'  Shaping adds dense, bounded signal without destabilizing.')
    else:
        print(f'WARNING — {(shape_abs > 5).mean()*100:.1f}% of steps have shaping |R| > 5.')


if __name__ == '__main__':
    main()
