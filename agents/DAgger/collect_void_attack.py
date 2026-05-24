"""Collect Void + Attack expert data, 2 parallel processes.

Void:  AlphaBeta d3 + aggressive ghosts → survival/interaction data
Attack: AlphaBeta d3 + coward ghosts → hunting data
"""
import sys, os, time, signal, numpy as np
from multiprocessing import Process

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL)

ACT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # placeholder, will import

def collect_worker(tag, ghost_configs, total_eps, output_file, depth=3):
    """Worker process: collect episodes with given ghost configs."""
    os.chdir(SKEL)

    import layout, ghostAgents
    from game import Directions
    from pacman import GameState
    from multiAgents import AlphaBetaAgent

    ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2,
           Directions.WEST: 3, Directions.STOP: 4}

    lo = layout.getLayout('mediumClassic')
    num_g = lo.getNumGhosts()

    # Walls grid
    H, W, C = 11, 20, 8
    wg = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        for y in range(H):
            if lo.walls.data[x][y]: wg[y, x] = 1.0

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
        g[7] = wg; return g

    agent = AlphaBetaAgent(depth=str(depth), evalFn='betterEvaluationFunction')
    all_trajs = []
    collected = 0
    t0 = time.time()

    for ghost_tag, attack, flee, count in ghost_configs:
        for _ in range(count):
            ghosts = [ghostAgents.DirectionalGhost(i+1, attack, flee) for i in range(num_g)]
            state = GameState(); state.initialize(lo, num_g)
            grids, actions = [], []
            step = 0

            while not (state.isWin() or state.isLose()) and step < 500:
                action = agent.getAction(state)
                grids.append(state_to_grid(state))
                actions.append(ACT.get(action, 4))
                state = state.generateSuccessor(0, action)
                if state.isWin() or state.isLose(): break
                for gi, g in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
                step += 1

            if len(grids) < 5: continue

            all_trajs.append({
                'states': np.array(grids, dtype=np.float32),
                'actions': np.array(actions, dtype=np.int32),
                'score': state.getScore(), 'win': state.isWin(),
                'steps': len(grids), 'ghost_profile': ghost_tag,
                'source': tag,
            })
            collected += 1

            if collected % 20 == 0:
                recent = [t['score'] for t in all_trajs[-20:]]
                print(f'[{tag}] {collected}/{total_eps} '
                      f'recent_avg={np.mean(recent):.0f} '
                      f'wins={sum(1 for t in all_trajs[-20:] if t["win"])} '
                      f'ghost={ghost_tag}')

    dt = time.time() - t0
    scores = [t['score'] for t in all_trajs]
    wins = sum(1 for t in all_trajs if t['win'])
    print(f'\n[{tag}] Done: {len(all_trajs)} eps in {dt:.0f}s '
          f'avg={np.mean(scores):.0f} wins={wins}/{len(all_trajs)}')

    np.savez_compressed(output_file, trajectories=np.array(all_trajs, dtype=object))
    print(f'[{tag}] Saved: {output_file}')


def main():
    # Void config: mostly aggressive → survival under pressure
    void_config = [
        ('aggressive', 0.9, 0.2, 100),
        ('balanced', 0.5, 0.5, 50),
    ]

    # Attack config: mostly coward → safe hunting practice
    attack_config = [
        ('coward', 0.2, 0.9, 80),
        ('balanced', 0.5, 0.5, 50),
        ('aggressive', 0.9, 0.2, 20),
    ]

    void_out = os.path.join(PROJECT, 'data', 'void_expert.npz')
    attack_out = os.path.join(PROJECT, 'data', 'attack_expert.npz')

    print(f'Void: 150 eps (mostly aggressive), AlphaBeta d3')
    print(f'Attack: 150 eps (mostly coward), AlphaBeta d3')
    print(f'Starting 2 parallel workers...\n')

    p1 = Process(target=collect_worker, args=('VOID', void_config, 150, void_out, 3))
    p2 = Process(target=collect_worker, args=('ATTACK', attack_config, 150, attack_out, 3))

    p1.start(); p2.start()
    p1.join(); p2.join()

    print('\nBoth workers done.')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
