"""Collect ghost-interaction data using Survival Expert.

Ghost profiles:
  - aggressive (0.9/0.2): 70% — maximum ghost pressure
  - balanced (0.5/0.5):   20% — baseline
  - coward (0.2/0.9):     10% — chasing practice

Agent: SurvivalAgent d2 (fast, ghost-avoidance specialist)
"""
import sys, os, time, signal, numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from game import Directions
from pacman import GameState
from survival_expert import SurvivalAgent, survivalEvaluation, _maze_dist

H, W, C = 11, 20, 8
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
       Directions.WEST: 3, Directions.STOP: 4}

# ── Walls ──
def get_walls():
    lo = layout.getLayout('mediumClassic')
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0
    return wg, lo

WG, LO = get_walls()

# ── State → Grid ──
def state_to_grid(state):
    g = np.zeros((C, H, W), dtype=np.float32)
    fd = state.getFood()
    for x in range(W):
        for y in range(H):
            if fd[x][y]: g[0, y, x] = 1.0
    for cx, cy in state.getCapsules():
        if 0 <= cx < W and 0 <= cy < H: g[1, cy, cx] = 1.0
    px, py = state.getPacmanPosition()
    if 0 <= px < W and 0 <= py < H: g[2, py, px] = 1.0
    for i, gh in enumerate(state.getGhostStates()):
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < W and 0 <= gy < H:
            g[3 + i, gy, gx] = 1.0
            g[5 + i, gy, gx] = gh.scaredTimer / 40.0
    g[7] = WG; return g

# ── Ghost distance (min distance to any active ghost) ──
def min_ghost_dist(state):
    pacman = state.getPacmanPosition()
    walls = state.getWalls()
    min_d = 999
    for gh in state.getGhostStates():
        if gh.scaredTimer == 0:
            d = _maze_dist(pacman, gh.getPosition(), walls)
            if d < min_d: min_d = d
    return min_d

# ── Run one episode ──
def run_episode(agent, ghost_factory, lo):
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    ghosts = ghost_factory(lo)
    grids, actions, scores = [], [], []
    prev_score = state.getScore()
    step = 0

    while not (state.isWin() or state.isLose()) and step < 500:
        action = agent.getAction(state)
        grids.append(state_to_grid(state))
        actions.append(ACT[action])
        state = state.generateSuccessor(0, action)
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi + 1, g.getAction(state) or Directions.STOP)
        scores.append(state.getScore() - prev_score)
        prev_score = state.getScore()
        step += 1

    if len(grids) < 5: return None
    return {'states': np.array(grids, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int32),
            'score': state.getScore(), 'win': state.isWin(),
            'steps': len(grids), 'ghost': 'survival'}

# ── Main ──
def main():
    agent = SurvivalAgent(depth=2)
    print(f'SurvivalAgent d2 loaded')

    ghost_configs = [
        ('aggressive', 0.9, 0.2, 70),   # 70 eps — maximum pressure
        ('balanced', 0.5, 0.5, 20),     # 20 eps — baseline
        ('coward', 0.2, 0.9, 10),       # 10 eps — chase practice
    ]

    all_trajs = []
    total_target = sum(c[3] for c in ghost_configs)
    collected = 0
    stopped = False

    def on_sig(sig, frame):
        nonlocal stopped; stopped = True
    signal.signal(signal.SIGINT, on_sig)

    print(f'Target: {total_target} episodes')
    t0 = time.time()

    for tag, attack, flee, count in ghost_configs:
        ghost_fac = lambda lo: [ghostAgents.DirectionalGhost(i+1, attack, flee)
                                 for i in range(lo.getNumGhosts())]

        for i in range(count):
            if stopped: break
            traj = run_episode(agent, ghost_fac, LO)
            if traj is None: continue
            traj['ghost_profile'] = tag
            all_trajs.append(traj)
            collected += 1

            if collected % 10 == 0:
                recent = [t['score'] for t in all_trajs[-10:]]
                print(f'[{collected:3d}/{total_target}] tag={tag} '
                      f'recent_avg={np.mean(recent):.0f} wins={sum(1 for t in all_trajs[-10:] if t["win"])}')

    dt = time.time() - t0
    scores = [t['score'] for t in all_trajs]
    wins = sum(1 for t in all_trajs if t['win'])

    print(f'\nDone: {len(all_trajs)} eps in {dt:.0f}s')
    print(f'  Avg score: {np.mean(scores):.0f}  Wins: {wins}/{len(all_trajs)}')

    out = os.path.join(PROJECT, 'data', 'survival_expert.npz')
    np.savez_compressed(out, trajectories=np.array(all_trajs, dtype=object))
    print(f'  Saved: {out}')

if __name__ == '__main__':
    main()
