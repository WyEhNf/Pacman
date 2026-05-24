"""Survival Expert — Ghost-avoidance specialist for data generation.

Uses alpha-beta search with a survival-focused evaluation function.
Designed to produce high-quality ghost-interaction trajectories.
"""
import sys, os
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKEL = os.path.join(PROJECT, 'PPCA-AIPacMan-2024-main', 'multiagent')
sys.path.insert(0, PROJECT); sys.path.insert(0, SKEL); os.chdir(SKEL)

from game import Directions, Agent
from pacman import GameState
from util import manhattanDistance, Queue

# ── Maze distance cache (same as betterEvaluationFunction) ──
_maze_cache = {}

def _maze_dist(pos1, pos2, walls):
    p1 = (int(pos1[0]), int(pos1[1]))
    p2 = (int(pos2[0]), int(pos2[1]))
    wid = id(walls)
    if wid not in _maze_cache:
        _maze_cache[wid] = {}
    cache = _maze_cache[wid]
    key = (p1, p2) if p1 <= p2 else (p2, p1)
    if key in cache:
        return cache[key]

    visited = {p1}
    q = Queue(); q.push((p1[0], p1[1], 0))
    while not q.isEmpty():
        x, y, d = q.pop()
        if (x, y) == p2:
            cache[key] = d; return d
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and not walls[nx][ny]:
                visited.add((nx, ny)); q.push((nx, ny, d+1))
    cache[key] = 999999; return 999999


class SurvivalAgent(Agent):
    """Alpha-beta agent with survival-prioritized evaluation.

    Focus: ghost avoidance >> food collection. Designed for aggressive ghosts.
    """

    def __init__(self, depth=2, evalFn=None):
        super().__init__(0)
        self.depth = int(depth)
        # Use survival_eval as default
        self.evaluationFunction = evalFn if evalFn else survivalEvaluation

    def getAction(self, gameState):
        best_score = -float('inf')
        best_action = Directions.STOP
        alpha, beta = -float('inf'), float('inf')

        legal = gameState.getLegalActions(0)
        if not legal:
            return Directions.STOP

        # Action ordering: safe actions first (away from ghosts)
        scored = [(self.evaluationFunction(gameState.generateSuccessor(0, a)), a)
                  for a in legal]
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, action in scored:
            successor = gameState.generateSuccessor(0, action)
            v = self._alpha_beta(successor, self.depth, 1, alpha, beta)
            if v > best_score:
                best_score = v; best_action = action
            alpha = max(alpha, best_score)
        return best_action

    def _alpha_beta(self, state, depth, agent_idx, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legal = state.getLegalActions(agent_idx)
        if not legal:
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        if agent_idx == num_agents - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agent_idx + 1, depth

        # Ghost ordering: worst-for-pacman first
        scored = [(self.evaluationFunction(state.generateSuccessor(agent_idx, a)), a)
                  for a in legal]

        if agent_idx == 0:
            scored.sort(key=lambda x: x[0], reverse=True)
            v = -float('inf')
            for _, a in scored:
                v = max(v, self._alpha_beta(
                    state.generateSuccessor(agent_idx, a), next_depth, next_agent, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v
        else:
            scored.sort(key=lambda x: x[0])
            v = float('inf')
            for _, a in scored:
                v = min(v, self._alpha_beta(
                    state.generateSuccessor(agent_idx, a), next_depth, next_agent, alpha, beta))
                if v < alpha: return v
                beta = min(beta, v)
            return v


def survivalEvaluation(gameState):
    """
    Ghost-survival-focused evaluation. Priorities:
      1. Stay alive (huge penalty for ghost proximity)
      2. Collect capsules (escape options = safety)
      3. Chase scared ghosts (safe points)
      4. Eat food (minor, to maintain realistic movement)
    """
    if gameState.isLose():
        return -10000.0
    if gameState.isWin():
        return 10000.0

    pacman_pos = gameState.getPacmanPosition()
    walls = gameState.getWalls()
    ghost_states = gameState.getGhostStates()
    capsules = gameState.getCapsules()
    food = gameState.getFood()
    food_list = food.asList()
    score = gameState.getScore()

    # ── Ghost Danger (DOMINANT) ──
    for ghost in ghost_states:
        gpos = ghost.getPosition()
        dist = _maze_dist(pacman_pos, gpos, walls) + 0.5

        if ghost.scaredTimer > 0:
            # Scared ghost: chase it (safe points)
            if dist < 8:
                score += 300.0 / (dist + 1.0)
                score += ghost.scaredTimer * 3.0
        else:
            # Active ghost: EXTREME danger penalty
            if dist < 3:
                score -= 2000.0 / (dist + 0.1)   # immediate death risk
            elif dist < 6:
                score -= 500.0 / (dist + 0.5)     # high danger zone
            elif dist < 10:
                score -= 100.0 / (dist + 1.0)     # caution zone
            else:
                score -= 20.0 / (dist + 1.0)      # mild awareness

    # ── Capsule attraction (safety resource) ──
    for cap_pos in capsules:
        dist = _maze_dist(pacman_pos, cap_pos, walls)
        if dist < 8:
            score += 150.0 / (dist + 1.0)  # nearby escape route
        else:
            score += 30.0 / (dist + 1.0)   # distant safety

    # ── Food (mild, maintain realistic movement) ──
    if food_list:
        nearest_food = min(_maze_dist(pacman_pos, f, walls) for f in food_list)
        score += 5.0 / (nearest_food + 1.0)
    score -= 1.0 * len(food_list)  # progress indicator

    # ── Safety metric: prefer open spaces ──
    # Count safe neighbors (not walls, far from ghosts)
    safe_neighbors = 0
    px, py = pacman_pos
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = px+dx, py+dy
        if not walls[nx][ny]:
            # Check ghost proximity
            min_ghost_dist = min(
                (_maze_dist((nx, ny), g.getPosition(), walls)
                 for g in ghost_states if g.scaredTimer == 0),
                default=99
            )
            if min_ghost_dist > 3:
                safe_neighbors += 1
    score += 20.0 * safe_neighbors  # escape options

    return score


# Abbreviation for multiAgents import
better = survivalEvaluation


# ── Test ──
if __name__ == '__main__':
    import layout, ghostAgents
    from game import Game
    from pacman import ClassicGameRules

    lo = layout.getLayout('mediumClassic')
    agent = SurvivalAgent(depth=2)

    # Aggressive ghosts for testing
    ghosts = [ghostAgents.DirectionalGhost(i+1, 0.9, 0.2) for i in range(lo.getNumGhosts())]

    print(f'SurvivalAgent d2 vs Aggressive Ghosts')
    print(f'Ghosts: attack=0.9, flee=0.2')

    # Headless run
    state = GameState(); state.initialize(lo, lo.getNumGhosts())
    step = 0
    while not (state.isWin() or state.isLose()) and step < 500:
        action = agent.getAction(state)
        state = state.generateSuccessor(0, action)
        if state.isWin() or state.isLose(): break
        for gi, g in enumerate(ghosts):
            if state.isWin() or state.isLose(): break
            state = state.generateSuccessor(gi+1, g.getAction(state) or Directions.STOP)
        step += 1

    print(f'Score: {state.getScore()}  Win: {state.isWin()}  Steps: {step}')
