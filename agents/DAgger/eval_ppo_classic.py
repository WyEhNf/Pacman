"""Evaluate PPO model on Berkeley classic layout."""
import sys, os, numpy as np, torch, time
from collections import deque

PROJECT = r'E:\Pacman'

# Import Pacman-AI (model + config)
sys.path.insert(0, os.path.join(PROJECT, 'pacman-ai-master'))
from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic

config = load_config()
env_cfg = config['env']; net_cfg = config['network']
fs = env_cfg.get('frame_stack', 4)

model = ActorCritic(
    grid_channels=env_cfg['observation_channels'] * fs,
    num_scalars=env_cfg.get('num_scalar_features', 5),
    cnn_channels=net_cfg['cnn_channels'], cnn_kernels=net_cfg['cnn_kernels'],
    cnn_strides=net_cfg['cnn_strides'], shared_hidden=net_cfg['shared_hidden'],
    head_hidden=net_cfg['head_hidden'],
)

ckpt = os.path.join(PROJECT, 'pacman-ai-master', 'runs', '2026-05-24_00-23-37', 'checkpoints', 'update_7999.pt')
model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
model.eval()

# Remove pacman-ai from path
sys.path.pop(0)
for k in list(sys.modules.keys()):
    if k.startswith('pacman'): del sys.modules[k]

# Import Berkeley
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, SKEL); os.chdir(SKEL)

import layout, ghostAgents
from pacman import GameState
from game import Directions

MAZE_ROWS, MAZE_COLS = 31, 28
ACT = {Directions.NORTH: 0, Directions.SOUTH: 1, Directions.EAST: 2, Directions.WEST: 3, Directions.STOP: 4}
REV = {0: Directions.NORTH, 1: Directions.SOUTH, 2: Directions.EAST, 3: Directions.WEST}

lo = layout.getLayout('classic')
num_ghosts = lo.numGhosts

def to_row(y): return MAZE_ROWS - 1 - y

def build_obs(state, prev_dir):
    """Exact pacman-ai format:
    Ch0=Walls, Ch1=Pac-Man, Ch2=Pellets, Ch3=Power pellets,
    Ch4=Dangerous ghosts, Ch5=Edible ghosts, Ch6=Ghost house, Ch7=Fruit
    """
    walls = state.getWalls(); food = state.getFood()
    grid = np.zeros((8, MAZE_ROWS, MAZE_COLS), dtype=np.float32)

    for x in range(min(MAZE_COLS, walls.width)):
        for y in range(min(MAZE_ROWS, walls.height)):
            r = to_row(y)
            if walls[x][y]: grid[0, r, x] = 1.0     # Ch0: Walls
            if food[x][y]: grid[2, r, x] = 1.0       # Ch2: Pellets

    # Ch1: Pac-Man
    px, py = state.getPacmanPosition()
    if 0 <= px < MAZE_COLS and 0 <= py < MAZE_ROWS:
        grid[1, to_row(py), px] = 1.0

    # Ch3: Power pellets
    for cx, cy in state.getCapsules():
        if 0 <= cx < MAZE_COLS and 0 <= cy < MAZE_ROWS:
            grid[3, to_row(cy), cx] = 1.0

    # Ch4: Dangerous ghosts (not scared), Ch5: Edible ghosts (scared)
    for gh in state.getGhostStates()[:4]:
        gx, gy = int(gh.getPosition()[0]), int(gh.getPosition()[1])
        if 0 <= gx < MAZE_COLS and 0 <= gy < MAZE_ROWS:
            r = to_row(gy)
            if gh.scaredTimer > 0:
                grid[5, r, gx] = 1.0   # Ch5: Edible
            else:
                grid[4, r, gx] = 1.0   # Ch4: Dangerous

    # Ch6: Ghost house (walls around ghost spawn area)
    # Approximate: cells near ghost start positions
    ghost_start_rows = [to_row(gy) for _, gy in [(8,5),(10,5),(11,5)]]

    # Ch7: Fruit (not implemented, leave empty)

    return grid

def legal_mask(state):
    legal = state.getLegalActions(0)
    mask = np.zeros(4, dtype=bool)
    for i, d in enumerate([Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]):
        if d in legal: mask[i] = True
    return mask

print(f'PPO (update 7999) on Berkeley classic: {MAZE_COLS}x{MAZE_ROWS}, {num_ghosts} ghosts, {lo.totalFood} food')
print(f'Ghosts: DirectionalGhost x{num_ghosts} (balanced 0.5/0.5)\n')

scores, wins = [], 0
t0 = time.time()
for ep in range(20):
    state = GameState(); state.initialize(lo, num_ghosts)
    ghosts = [ghostAgents.DirectionalGhost(i + 1, 0.5, 0.5) for i in range(num_ghosts)]
    frame_buf = deque(maxlen=fs)
    prev_dir = None
    step = 0

    # Init frame buffer
    g0 = build_obs(state, prev_dir)
    for _ in range(fs): frame_buf.append(g0.copy())

    while not (state.isWin() or state.isLose()) and step < 3000:
        stacked = np.concatenate(list(frame_buf), axis=0)
        # Scalars: power_timer, lives_frac, ghosts_eaten, pellets_eaten, pac_dir
        pel_eaten = lo.totalFood - state.getFood().count() if hasattr(state.getFood(), 'count') else 0.5
        shy_ghost = [g for g in state.getGhostStates() if g.scaredTimer > 0]
        power_t = max(g.scaredTimer for g in state.getGhostStates()) / 40.0 if state.getGhostStates() else 0.0
        scalars = np.array([power_t, 3.0/3.0, min(len(shy_ghost)/4.0, 1.0),
                           float(lo.totalFood - state.getNumFood()) / lo.totalFood,
                           prev_dir / 3.0 if prev_dir is not None else 0.5], dtype=np.float32)
        mask = legal_mask(state)

        with torch.no_grad():
            g = torch.FloatTensor(stacked).unsqueeze(0)
            s = torch.FloatTensor(scalars).unsqueeze(0)
            m = torch.BoolTensor(mask).unsqueeze(0)
            logits, _ = model(g, s, m)
            probs = torch.softmax(logits, -1)[0]
            action_idx = probs.argmax().item()

        # Take action
        action = REV.get(action_idx, Directions.STOP)
        prev_dir = action_idx
        state = state.generateSuccessor(0, action)

        if state.isWin() or state.isLose(): break
        for gi, gh in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi + 1, gh.getAction(state) or Directions.STOP)

        frame_buf.append(build_obs(state, prev_dir))
        step += 1

    scores.append(state.getScore())
    if state.isWin(): wins += 1
    print(f'Ep{ep:2d}: {scores[-1]:6.0f}  {"WIN" if state.isWin() else ""}  ({step} steps)')

dt = time.time() - t0
print(f'\nAvg={np.mean(scores):.0f}  Wins={wins}/20  Min={min(scores):.0f}  Max={max(scores):.0f}  [{dt:.0f}s]')
