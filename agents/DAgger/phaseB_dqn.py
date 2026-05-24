"""Phase B: Double DQN fine-tuning from DAgger_R1 with reward shaping.

- Double DQN (online + target network)
- Reward = game_score_delta + Phase A shaping
- Mixed replay: 70% uniform + 20% death frames + 10% kill frames
- Epsilon-greedy: 0.3 → 0.05 over 5000 steps
- Self-play on mediumClassic with varied ghost profiles

Usage:
    python scripts/phaseB_dqn.py --steps 50000
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import random, time
from collections import deque

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

# ── Layout & walls ──
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

# ── Reward shaping (from Phase A) ──
def nearest_food_dist(state, px, py):
    fd = state.getFood(); best = 999
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
    return min(abs(px - cx) + abs(py - cy) for cx, cy in caps)

def any_nonscared_ghost_near(state, px, py, threshold=6):
    for g in state.getGhostStates():
        if g.scaredTimer <= 0:
            gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
            if abs(px - gx) + abs(py - gy) <= threshold:
                return True
    return False

class RewardShaper:
    """Tracks per-episode state for distance-based rewards."""
    def reset(self, state):
        px, py = state.getPacmanPosition()
        self.prev_food_dist = nearest_food_dist(state, px, py)
        self.prev_capsule_dist = nearest_capsule_dist(state, px, py)
        self.prev_scared_dist = nearest_scared_ghost_dist(state, px, py)
        self.prev_ghost_dist = min_ghost_dist(state, px, py)
        self.prev_score = state.getScore()
        self.prev_food_count = state.getFood().count()

    def compute(self, state, action_dir, prev_dir):
        px, py = state.getPacmanPosition()

        # R_base
        curr_score = state.getScore()
        R_base = curr_score - self.prev_score
        self.prev_score = curr_score

        # R_danger
        ghost_dist = min_ghost_dist(state, px, py)
        if ghost_dist <= 2:       R_danger = -3.0
        elif ghost_dist <= 4:     R_danger = -1.0
        elif ghost_dist <= 6:     R_danger = -0.3
        else:                      R_danger = 0.0

        # R_death
        R_death = -500.0 if state.isLose() else 0.0

        # R_food_nav
        food_dist = nearest_food_dist(state, px, py)
        R_food_nav = 0.0
        if self.prev_food_dist < 999 and food_dist < 999:
            R_food_nav = np.clip(0.3 * (self.prev_food_dist - food_dist), -3.0, 3.0)
        self.prev_food_dist = food_dist

        # R_food_eaten
        curr_food_count = state.getFood().count()
        food_eaten = self.prev_food_count - curr_food_count
        R_food_eaten = 2.0 * food_eaten if food_eaten > 0 else 0.0
        self.prev_food_count = curr_food_count

        # Track if a ghost was killed (score jump of ~200)
        ghost_killed = (R_base >= 150)  # ghost kill = +200, capsule = +50

        # R_capsule
        capsule_dist = nearest_capsule_dist(state, px, py)
        R_capsule = 0.0
        if any_nonscared_ghost_near(state, px, py, 6) and capsule_dist < 999:
            if self.prev_capsule_dist < 999 and capsule_dist < 999:
                R_capsule = np.clip(1.0 * (self.prev_capsule_dist - capsule_dist), -3.0, 3.0)
        self.prev_capsule_dist = capsule_dist

        # R_chase
        scared_dist = nearest_scared_ghost_dist(state, px, py)
        R_chase = 0.0
        if scared_dist < 999:
            if self.prev_scared_dist < 999 and scared_dist < 999:
                R_chase = np.clip(1.5 * (self.prev_scared_dist - scared_dist), -3.0, 3.0)
            elif self.prev_scared_dist >= 999:
                R_chase = 1.5
        self.prev_scared_dist = scared_dist

        # R_momentum
        R_momentum = 0.1 if (prev_dir and action_dir == prev_dir) else 0.0

        # R_win
        R_win = 200.0 if state.isWin() else 0.0

        # R_time
        R_time = -0.05

        total = (R_base + R_danger + R_death + R_food_nav + R_food_eaten +
                 R_capsule + R_chase + R_momentum + R_win + R_time)

        info = {
            'R_base': R_base, 'R_danger': R_danger, 'R_death': R_death,
            'R_food_nav': R_food_nav, 'R_food_eaten': R_food_eaten,
            'R_capsule': R_capsule, 'R_chase': R_chase,
            'R_momentum': R_momentum, 'R_win': R_win, 'R_time': R_time,
            'ghost_killed': ghost_killed,
        }
        return total, info

# ── Ghosts ──
GHOST_PROFILES = {
    'balanced':   (0.5, 0.5),
    'aggressive': (0.9, 0.2),
    'coward':     (0.2, 0.9),
    'random':     None,
}

def make_ghosts(profile):
    if profile == 'random':
        return [ghostAgents.RandomGhost(i + 1) for i in range(lo.getNumGhosts())]
    a, f = GHOST_PROFILES[profile]
    return [ghostAgents.DirectionalGhost(i + 1, a, f) for i in range(lo.getNumGhosts())]

GHOST_WEIGHTS = [0.5, 0.2, 0.15, 0.15]  # balanced, aggressive, coward, random

# ── Replay Buffer ──
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.death_buffer = deque(maxlen=5000)
        self.kill_buffer = deque(maxlen=5000)

    def push(self, s, a, r, s_next, done, is_death, is_kill):
        entry = (s, a, r, s_next, done)
        self.buffer.append(entry)
        if is_death:
            self.death_buffer.append(entry)
        if is_kill:
            self.kill_buffer.append(entry)

    def sample(self, batch_size):
        n_kill = min(int(batch_size * 0.1), len(self.kill_buffer))
        n_death = min(int(batch_size * 0.2), len(self.death_buffer))
        n_main = batch_size - n_kill - n_death

        batches = []
        if n_main > 0:
            batches.extend(random.sample(list(self.buffer), min(n_main, len(self.buffer))))
        if n_death > 0:
            batches.extend(random.sample(list(self.death_buffer), n_death))
        if n_kill > 0:
            batches.extend(random.sample(list(self.kill_buffer), n_kill))

        # Unpack
        s = np.stack([b[0] for b in batches])
        a = np.array([b[1] for b in batches], dtype=np.int64)
        r = np.array([b[2] for b in batches], dtype=np.float32)
        s_next = np.stack([b[3] for b in batches])
        done = np.array([b[4] for b in batches], dtype=np.float32)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)

# ── Training ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    parser.add_argument('--eps_start', type=float, default=0.3)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--eval_eps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load pretrained DAgger R1 ──
    print('Loading DAgger R1...')
    base_state = torch.load(os.path.join(PROJECT, 'checkpoints/dagger_cnn_m0_final.pt'), map_location=device)

    q_net = CNNDQN().to(device)
    q_net.load_state_dict(base_state)

    target_net = CNNDQN().to(device)
    target_net.load_state_dict(base_state)
    target_net.eval()

    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr, weight_decay=1e-5)
    buffer = ReplayBuffer(capacity=args.buffer_capacity)
    shaper = RewardShaper()

    # ── Stats ──
    global_step = 0
    episode = 0
    ep_scores, ep_steps, ep_food_pct, ep_deaths = [], [], [], []
    losses = deque(maxlen=200)
    q_means = deque(maxlen=200)
    t0 = time.time()

    def eval_model(n_eps=20):
        """Evaluate without exploration."""
        scores, wins, steps_arr, food_pcts = [], 0, [], []
        for ep_i in range(n_eps):
            profile = random.choice(list(GHOST_PROFILES.keys()))
            ghosts = make_ghosts(profile)
            state = GameState(); state.initialize(lo, lo.getNumGhosts())
            step = 0
            while not (state.isWin() or state.isLose()) and step < 500:
                g = state_to_grid(state)
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(g).unsqueeze(0).to(device))[0].cpu().numpy()
                legal = state.getLegalActions(0)
                ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
                if not ids: ids = [4]
                best, mv = -1e9, 4
                for i in range(5):
                    if i in ids and q[i] > best: best = q[i]; mv = i
                action_dir = REV[mv]
                state = state.generateSuccessor(0, action_dir)
                if state.isWin() or state.isLose(): break
                for gi, gs in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)
                step += 1
            scores.append(state.getScore())
            if state.isWin(): wins += 1
            steps_arr.append(step)
            total_food = lo.totalFood
            eaten = total_food - state.getFood().count()
            food_pcts.append(eaten / total_food * 100 if total_food > 0 else 0)
        return np.mean(scores), wins / n_eps, np.mean(steps_arr), np.mean(food_pcts)

    print(f'\n{"="*60}')
    print(f'Phase B: Double DQN Fine-tuning')
    print(f'  Steps: {args.steps}  LR: {args.lr}  Batch: {args.batch}')
    print(f'  Epsilon: {args.eps_start} → {args.eps_end} over {args.eps_decay} steps')
    print(f'  Target update: every {args.target_update} steps')
    print(f'  Train freq: every {args.train_freq} steps')
    print(f'{"="*60}\n')

    # Initial eval
    avg_s, wr, avg_st, avg_food = eval_model(args.eval_eps)
    print(f'[INITIAL] score={avg_s:.0f}  win={wr:.1%}  steps={avg_st:.0f}  food%={avg_food:.0f}%\n')

    best_avg_score = avg_s
    best_path = os.path.join(PROJECT, 'checkpoints', 'phaseB_dqn_best.pt')

    while global_step < args.steps:
        # ── Collect one episode ──
        eps = args.eps_start + (args.eps_end - args.eps_start) * min(1.0, global_step / args.eps_decay)
        profile = random.choices(list(GHOST_PROFILES.keys()), weights=GHOST_WEIGHTS, k=1)[0]
        ghosts = make_ghosts(profile)
        state = GameState(); state.initialize(lo, lo.getNumGhosts())
        shaper.reset(state)

        prev_dir = None
        ep_reward = 0
        ep_step = 0
        ep_kills = 0
        ep_losses = []

        while not (state.isWin() or state.isLose()) and ep_step < 500:
            # s_t: state before any agent moves
            grid = state_to_grid(state)

            # Epsilon-greedy
            if random.random() < eps:
                legal = state.getLegalActions(0)
                legal = [a for a in legal if a != Directions.STOP] or legal
                action_dir = random.choice(legal)
                mv = ACT[action_dir]
            else:
                with torch.no_grad():
                    q = q_net(torch.FloatTensor(grid).unsqueeze(0).to(device))[0].cpu().numpy()
                legal = state.getLegalActions(0)
                ids = [ACT[a] for a in legal if a != Directions.STOP or len(legal) == 1]
                if not ids: ids = [4]
                best, mv = -1e9, 4
                for i in range(5):
                    if i in ids and q[i] > best: best = q[i]; mv = i
                action_dir = REV[mv]

            # Pacman moves
            state = state.generateSuccessor(0, action_dir)

            # Reward (after pacman move, before ghosts — captures food/capsule/ghost events)
            R, r_info = shaper.compute(state, action_dir, prev_dir)
            ep_reward += R
            is_kill = r_info['ghost_killed']
            if is_kill: ep_kills += 1

            # Ghosts move → s_{t+1}
            pacman_died = state.isLose()  # died during pacman's own move
            if not (state.isWin() or state.isLose()):
                for gi, gs in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi + 1, gs.getAction(state) or Directions.STOP)

            # s_{t+1} grid (after all agents moved)
            done = state.isWin() or state.isLose()
            if done:
                next_grid = np.zeros((C, H, W), dtype=np.float32)
                # If ghost killed pacman during ghost phase, capture death reward
                if state.isLose() and not pacman_died:
                    R -= 500.0  # append death penalty to this transition's reward
            else:
                next_grid = state_to_grid(state)

            is_death = state.isLose()

            # Push to replay buffer
            buffer.push(grid, mv, R, next_grid, done, is_death, is_kill)
            prev_dir = action_dir
            ep_step += 1
            global_step += 1

            # ── Train ──
            if global_step % args.train_freq == 0 and len(buffer) >= args.batch:
                s_b, a_b, r_b, sn_b, d_b = buffer.sample(args.batch)
                s_t = torch.FloatTensor(s_b).to(device)
                a_t = torch.LongTensor(a_b).unsqueeze(1).to(device)
                r_t = torch.FloatTensor(r_b).unsqueeze(1).to(device)
                sn_t = torch.FloatTensor(sn_b).to(device)
                d_t = torch.FloatTensor(d_b).unsqueeze(1).to(device)

                # Double DQN: online picks action, target evaluates
                with torch.no_grad():
                    q_next_online = q_net(sn_t)
                    best_actions = q_next_online.argmax(dim=1, keepdim=True)
                    q_next_target = target_net(sn_t).gather(1, best_actions)
                    target = r_t + args.gamma * q_next_target * (1 - d_t)

                q_current = q_net(s_t).gather(1, a_t)
                loss = F.smooth_l1_loss(q_current, target)

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()

                losses.append(loss.item())
                q_means.append(q_current.mean().item())
                ep_losses.append(loss.item())

            # Update target network
            if global_step % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if global_step >= args.steps:
                break

        # ── End of episode ──
        ep_scores.append(state.getScore())
        ep_steps.append(ep_step)
        total_food = lo.totalFood
        eaten = total_food - state.getFood().count()
        ep_food_pct.append(eaten / total_food * 100 if total_food > 0 else 0)
        ep_deaths.append(1 if state.isLose() else 0)
        episode += 1

        # ── Logging ──
        if episode % 10 == 0:
            r20 = np.mean(ep_scores[-10:])
            d20 = np.mean(ep_deaths[-10:])
            f20 = np.mean(ep_food_pct[-10:])
            l20 = np.mean(losses) if losses else 0
            q20 = np.mean(q_means) if q_means else 0
            eta = (time.time() - t0) / global_step * (args.steps - global_step) if global_step > 0 else 0
            print(f'[{global_step:6d}/{args.steps}] eps={eps:.3f} '
                  f'score10={r20:7.0f} death10={d20:.2f} food10={f20:.0f}% '
                  f'loss={l20:.4f} Q={q20:.2f} kills={ep_kills} '
                  f'buf={len(buffer):6d} eta={eta:.0f}s')

        # ── Evaluation ──
        if global_step % args.eval_interval == 0 and global_step > 0:
            avg_s, wr, avg_st, avg_food = eval_model(args.eval_eps)
            print(f'\n  === EVAL @ {global_step} steps ===')
            print(f'  score={avg_s:.0f}  win={wr:.1%}  steps={avg_st:.0f}  food%={avg_food:.0f}%\n')
            if avg_s > best_avg_score:
                best_avg_score = avg_s
                torch.save(q_net.state_dict(), best_path)
                print(f'  [Best model saved: {best_path}]\n')

    # ── Final ──
    torch.save(q_net.state_dict(), os.path.join(PROJECT, 'checkpoints', 'phaseB_dqn_final.pt'))
    print(f'\n{"="*60}')
    print(f'PHASE B COMPLETE')
    print(f'  Total steps:    {global_step}')
    print(f'  Episodes:       {episode}')
    print(f'  Best eval avg:  {best_avg_score:.0f}')
    print(f'  Final eps:      {eps:.3f}')

    # Final evaluation
    avg_s, wr, avg_st, avg_food = eval_model(50)
    print(f'\n  Final (50 eps):')
    print(f'    Avg score:    {avg_s:.0f}')
    print(f'    Win rate:     {wr:.1%}')
    print(f'    Avg steps:    {avg_st:.0f}')
    print(f'    Avg food%:    {avg_food:.0f}%')


if __name__ == '__main__':
    main()
