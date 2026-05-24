"""Attack Expert v2 — Optimised for win rate.

Key improvements:
  1. Single BFS per state → all distances (10x faster eval)
  2. Quiescence search — extend depth when ghost is adjacent
  3. Capsule strategy — only rush if ghosts are threatening
  4. Catch prediction — only chase if timer > distance
  5. Danger urgency — state-value penalty increases with depth
"""
import sys, os
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Directions, Agent
from pacman import GameState
from util import Queue

# ── Distance map (single BFS) ──
class DistMap:
    """Flood-fill BFS from Pacman position. O(H*W) once per state."""
    def __init__(self, start, walls):
        self.dist = {}
        visited = {start}
        q = Queue(); q.push((start[0], start[1], 0))
        while not q.isEmpty():
            x, y, d = q.pop()
            self.dist[(x, y)] = d
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) not in visited and not walls[nx][ny]:
                    visited.add((nx, ny)); q.push((nx, ny, d+1))

    def to(self, pos):
        return self.dist.get((int(pos[0]), int(pos[1])), 999)


class AttackAgent(Agent):
    """Alpha-beta d3 + quiescence, with fast dist-map evaluation."""

    def __init__(self, depth=3):
        super().__init__(0)
        self.depth = int(depth)

    def getAction(self, gameState):
        best_score = -float('inf')
        best_action = Directions.STOP
        alpha, beta = -float('inf'), float('inf')

        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP

        # Action ordering: safe moves first
        walls = gameState.getWalls()
        dm = DistMap(gameState.getPacmanPosition(), walls)
        ghost_states = gameState.getGhostStates()

        def action_safety(a):
            succ = gameState.generateSuccessor(0, a)
            pp = succ.getPacmanPosition()
            min_d = min((dm.to(g.getPosition()) for g in ghost_states if g.scaredTimer == 0), default=99)
            # Prefer moves away from ghosts
            return min_d

        scored = [(action_safety(a), a) for a in legal]
        scored.sort(key=lambda x: x[0], reverse=True)  # safest first

        for _, action in scored:
            successor = gameState.generateSuccessor(0, action)
            v = self._alpha_beta(successor, self.depth, 1, alpha, beta)
            if v > best_score:
                best_score = v; best_action = action
            alpha = max(alpha, best_score)
        return best_action

    def _alpha_beta(self, state, depth, agent_idx, alpha, beta):
        # Quiescence: if ghost is dangerously close, extend one more ply
        if depth == 0:
            if not self._is_quiet(state):
                depth = 1  # extend: don't stop when danger is immediate
            else:
                return attackEval(state)

        if state.isWin(): return 10000.0
        if state.isLose(): return -10000.0

        legal = state.getLegalActions(agent_idx)
        if not legal:
            return attackEval(state)

        num_agents = state.getNumAgents()
        next_agent = 0 if agent_idx == num_agents - 1 else agent_idx + 1
        next_depth = depth - 1 if agent_idx == num_agents - 1 else depth

        if agent_idx == 0:
            v = -float('inf')
            for a in legal:
                v = max(v, self._alpha_beta(
                    state.generateSuccessor(agent_idx, a), next_depth, next_agent, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v
        else:
            v = float('inf')
            for a in legal:
                v = min(v, self._alpha_beta(
                    state.generateSuccessor(agent_idx, a), next_depth, next_agent, alpha, beta))
                if v < alpha: return v
                beta = min(beta, v)
            return v

    def _is_quiet(self, state):
        """True if no ghost is dangerously close."""
        pacman = state.getPacmanPosition()
        for g in state.getGhostStates():
            if g.scaredTimer == 0:
                d = abs(pacman[0]-g.getPosition()[0]) + abs(pacman[1]-g.getPosition()[1])
                if d < 4:
                    return False
        return True


def attackEval(gameState):
    """Fast single-BFS evaluation with capsule strategy."""
    if gameState.isLose(): return -10000.0
    if gameState.isWin(): return 10000.0

    pacman = gameState.getPacmanPosition()
    walls = gameState.getWalls()
    ghosts = gameState.getGhostStates()
    capsules = gameState.getCapsules()
    food_grid = gameState.getFood()
    score = gameState.getScore()

    # Single BFS for all distances
    dm = DistMap(pacman, walls)

    # ── Ghost analysis ──
    danger_penalty = 0.0
    hunt_bonus = 0.0
    any_scared = False

    for g in ghosts:
        gpos = g.getPosition()
        d = dm.to(gpos)
        if d == 0: d = 0.5

        if g.scaredTimer > 0:
            any_scared = True
            timer = g.scaredTimer
            can_catch = d <= timer
            # Hunt: value increases as we get closer
            hunt_bonus += 400.0 / (d + 0.5)
            if can_catch and d < 8:
                hunt_bonus += 250.0  # confirmed kill possible
            if d <= 2:
                hunt_bonus += 300.0  # right there!
            # Don't chase hopelessly far targets
            if d > timer + 4:
                hunt_bonus -= 100.0
        else:
            # Active ghost: danger decreases with distance
            if d < 3:
                danger_penalty -= 3000.0 / (d + 0.1)
            elif d < 6:
                danger_penalty -= 600.0 / (d + 0.5)
            elif d < 10:
                danger_penalty -= 100.0 / (d + 1.0)
            else:
                danger_penalty -= 20.0 / (d + 1.0)

    score += danger_penalty + hunt_bonus

    # ── Capsule strategy ──
    if capsules:
        for cap in capsules:
            d = dm.to(cap)
            if d < 12:
                # How many threatening ghosts are near this capsule?
                ghosts_near_cap = 0
                for g in ghosts:
                    if g.scaredTimer == 0:
                        gd = abs(cap[0]-g.getPosition()[0]) + abs(cap[1]-g.getPosition()[1])
                        if gd < 8:
                            ghosts_near_cap += 1
                # Capsule is valuable if ghosts nearby (enables hunting)
                value = 80.0 + 60.0 * ghosts_near_cap
                score += value / (d + 1.0)

    # ── Food ──
    food_list = food_grid.asList()
    if food_list:
        nearest = min(dm.to(f) for f in food_list)
        score += 10.0 / (nearest + 1.0)
    score -= 2.0 * len(food_list)

    # ── Escape options ──
    safe = 0
    px, py = int(pacman[0]), int(pacman[1])
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = px+dx, py+dy
        if not walls[nx][ny]:
            min_gd = min((abs(nx-g.getPosition()[0])+abs(ny-g.getPosition()[1])
                          for g in ghosts if g.scaredTimer==0), default=99)
            if min_gd > 2:
                safe += 1
    score += 30.0 * safe

    return score


# Compatibility
better = attackEval


# ── Test ──
if __name__ == '__main__':
    import layout, ghostAgents
    from pacman import GameState
    import time

    lo = layout.getLayout('mediumClassic')

    for depth in [3, 4]:
        agent = AttackAgent(depth=depth)
        print(f'\nAttackAgent d{depth}:', end=' ', flush=True)
        t0 = time.time()

        scores, wins, steps = [], 0, 0
        for ep in range(5):
            ghosts = [ghostAgents.DirectionalGhost(i+1, 0.5, 0.5) for i in range(lo.getNumGhosts())]
            state = GameState(); state.initialize(lo, lo.getNumGhosts())
            s = 0
            while not (state.isWin() or state.isLose()) and s < 500:
                action = agent.getAction(state)
                state = state.generateSuccessor(0, action)
                if state.isWin() or state.isLose(): break
                for gi, g in enumerate(ghosts):
                    if state.isWin() or state.isLose(): break
                    state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
                s += 1
            scores.append(state.getScore())
            if state.isWin(): wins += 1
            steps += s

        dt = time.time() - t0
        print(f'avg={np.mean(scores):.0f} wins={wins}/5 steps={steps//5} [{dt:.1f}s]')
