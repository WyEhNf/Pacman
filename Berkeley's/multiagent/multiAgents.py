# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        foodList = newFood.asList()
        capsules = successorGameState.getCapsules()

        # Distance to nearest food (closer is better; avoid division by zero)
        if foodList:
            nearestFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            score += 1.0 / nearestFoodDist if nearestFoodDist > 0 else 10.0

        # Ghost handling
        scaredGhostDist = float('inf')
        dangerousGhostDist = float('inf')

        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(newPos, ghost.getPosition())
            if dist == 0:
                dist = 0.5  # Prevent division by zero
            if scaredTime > 0:
                # Scared ghost: chase it
                if dist < scaredGhostDist:
                    scaredGhostDist = dist
            else:
                # Dangerous ghost: stay away
                if dist < dangerousGhostDist:
                    dangerousGhostDist = dist

        # Reward being close to scared ghosts
        if scaredGhostDist != float('inf'):
            score += 20.0 / scaredGhostDist

        # Penalty for being close to dangerous ghosts
        if dangerousGhostDist != float('inf'):
            score -= 10.0 / dangerousGhostDist

        # Distance to nearest capsule (encourage eating power pellets)
        if capsules:
            nearestCapsuleDist = min(manhattanDistance(newPos, c) for c in capsules)
            score += 2.0 / nearestCapsuleDist if nearestCapsuleDist > 0 else 20.0

        # Bonus for eating food (fewer food remaining = better progress)
        score -= 3.0 * len(foodList)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        bestScore = -float('inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self._minimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def _minimax(self, state, depth, agentIndex):
        # Terminal or depth limit reached: evaluate the state
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)

        # Determine next agent and remaining depth
        if agentIndex == state.getNumAgents() - 1:
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth

        successors = [state.generateSuccessor(agentIndex, a) for a in legalActions]
        values = [self._minimax(s, nextDepth, nextAgent) for s in successors]

        if agentIndex == 0:
            return max(values)   # Pacman maximizes
        else:
            return min(values)   # Ghosts minimize

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestScore = -float('inf')
        bestAction = Directions.STOP
        alpha = -float('inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self._alphabeta(successor, self.depth, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction

    def _alphabeta(self, state, depth, agentIndex, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)

        if agentIndex == state.getNumAgents() - 1:
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth

        # Action ordering: evaluate 1-ply successors, sort to maximize pruning
        scored = [(self.evaluationFunction(state.generateSuccessor(agentIndex, a)), a)
                  for a in legalActions]
        if agentIndex == 0:
            # Pacman (MAX): best first → quickly raise alpha → prune more
            scored.sort(key=lambda x: x[0], reverse=True)
            v = -float('inf')
            for _, a in scored:
                successor = state.generateSuccessor(agentIndex, a)
                v = max(v, self._alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            # Ghost (MIN): worst first (for MAX) → quickly lower beta → prune more
            scored.sort(key=lambda x: x[0])
            v = float('inf')
            for _, a in scored:
                successor = state.generateSuccessor(agentIndex, a)
                v = min(v, self._alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        bestScore = -float('inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self._expectimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def _expectimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(state)

        if agentIndex == state.getNumAgents() - 1:
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth

        if agentIndex == 0:
            # Pacman (MAX)
            return max(self._expectimax(
                state.generateSuccessor(0, a), nextDepth, nextAgent)
                for a in legalActions)
        else:
            # Ghost (EXPECTATION over uniform random choices)
            values = [self._expectimax(
                state.generateSuccessor(agentIndex, a), nextDepth, nextAgent)
                for a in legalActions]
            return sum(values) / len(values)

# Cache: maze BFS distance lookup table — computed once per layout, used forever
_mazeDistanceCache = {}  # key: id(walls) → {pos: {pos: distance}}

def _mazeDist(pos1, pos2, walls):
    """BFS maze distance between two positions. Uses global cache."""
    wid = id(walls)
    if wid not in _mazeDistanceCache:
        _mazeDistanceCache[wid] = {}
    cache = _mazeDistanceCache[wid]
    key = (pos1, pos2) if pos1 <= pos2 else (pos2, pos1)
    if key in cache:
        return cache[key]

    # BFS from pos1 to pos2
    from util import Queue
    visited = {pos1}
    q = Queue(); q.push((pos1[0], pos1[1], 0))
    while not q.isEmpty():
        x, y, d = q.pop()
        if (x, y) == pos2:
            cache[key] = d
            return d
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and not walls[nx][ny]:
                visited.add((nx, ny))
                q.push((nx, ny, d+1))
    cache[key] = 999999  # unreachable
    return 999999


def betterEvaluationFunction(currentGameState: GameState):
    """
    Extreme ghost-hunting evaluation. Uses MAZE distance (BFS) for accuracy.
    Precomputed with cache — first evaluation of a layout is slower,
    subsequent calls are O(1) lookups per pair.
    """
    score = currentGameState.getScore()
    pacman_pos = currentGameState.getPacmanPosition()
    walls = currentGameState.getWalls()
    food = currentGameState.getFood()
    food_list = food.asList()
    capsules = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()

    # Food: maze distance
    if food_list:
        min_food_dist = min(_mazeDist(pacman_pos, f, walls) for f in food_list)
        score += 10.0 / (min_food_dist + 1)

    # Food remaining penalty
    score -= 4.0 * len(food_list)

    # Capsules: maze distance
    if capsules:
        min_cap_dist = min(_mazeDist(pacman_pos, c, walls) for c in capsules)
        score += 20.0 / (min_cap_dist + 1)

    # Ghosts
    for ghost in ghost_states:
        gpos = ghost.getPosition()
        dist = _mazeDist(pacman_pos, gpos, walls) + 0.5

        if ghost.scaredTimer > 0:
            score += 200.0 / (dist + 1)
            score += ghost.scaredTimer * 2
        else:
            if dist < 5:
                score -= 500.0 / (dist + 0.1)
            else:
                score -= 50.0 / (dist + 1)

    return score

# Abbreviation
better = betterEvaluationFunction
