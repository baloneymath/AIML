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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        if action == 'Stop':
            return -1e7
        eat_food = 1e3
        eat_capsule = 2e3
        eaten_by_ghost = -1e9

        newx, newy = newPos
        newFoodList = newFood.asList()
        newGhostPosList = successorGameState.getGhostPositions()
        capsulesPosList = currentGameState.getCapsules()
        
        for i in range(len(newGhostPosList)):
            if manhattanDistance(newPos, newGhostPosList[i]) <= 1 and newScaredTimes[i] == 0:
                return eaten_by_ghost
        score = 0
        scaredGhostPosList = []
        for i in range(len(newGhostPosList)):
            if newScaredTimes[i] != 0:
                scaredGhostPosList.append(newGhostPosList[i])
        if len(scaredGhostPosList) != 0:
            score += max([1.0 / manhattanDistance(newPos, gPos) for gPos in scaredGhostPosList])
        if currentGameState.hasFood(newx, newy):
            score += eat_food
        if newPos in capsulesPosList:
            score += eat_capsule
        if len(newFoodList) != 0:
            inv_dist = [1.0 / manhattanDistance(newPos, food) for food in newFoodList]
            score += max(inv_dist)
            score += sum(inv_dist) / len(inv_dist)

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            next_agentIndex = agentIndex + 1
            if next_agentIndex == gameState.getNumAgents():
                next_agentIndex = 0
                depth -= 1
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = []
            if agentIndex == 0:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    scores.append(minimax(child, depth, next_agentIndex))
                return max(scores)
            else:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    scores.append(minimax(child, depth, next_agentIndex))
                return min(scores)

        legalMoves = gameState.getLegalActions(0)
        while legalMoves.count('Stop') != 0:
            legalMoves.remove('Stop')
        scores = [minimax(gameState.generateSuccessor(0, action), self.depth, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
        return legalMoves[random.choice(bestIndices)]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            next_agentIndex = agentIndex + 1
            if next_agentIndex == gameState.getNumAgents():
                next_agentIndex = 0
                depth -= 1
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = []
            if agentIndex == 0:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    score = alphabeta(child, depth, next_agentIndex, alpha, beta)
                    scores.append(score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                return max(scores)
            else:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    score = alphabeta(child, depth, next_agentIndex, alpha, beta)
                    scores.append(score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                return min(scores)

        legalMoves = gameState.getLegalActions(0)
        while legalMoves.count('Stop') != 0:
            legalMoves.remove('Stop')
        scores = [alphabeta(gameState.generateSuccessor(0, action), self.depth, 1, -1e9, 1e9) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
        return legalMoves[random.choice(bestIndices)]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMinimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            next_agentIndex = agentIndex + 1
            if next_agentIndex == gameState.getNumAgents():
                next_agentIndex = 0
                depth -= 1
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = []
            if agentIndex == 0:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    scores.append(expectiMinimax(child, depth, next_agentIndex))
                return max(scores)
            else:
                for action in legalMoves:
                    if action == 'Stop':
                        continue
                    child = gameState.generateSuccessor(agentIndex, action)
                    scores.append(expectiMinimax(child, depth, next_agentIndex))
                return float(sum(scores) / len(scores))

        legalMoves = gameState.getLegalActions(0)
        while legalMoves.count('Stop') != 0:
            legalMoves.remove('Stop')
        scores = [expectiMinimax(gameState.generateSuccessor(0, action), self.depth, 1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
        return legalMoves[random.choice(bestIndices)]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 1e9
    elif currentGameState.isLose():
        return -1e9
    walls = currentGameState.getWalls()
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostPositions()
    scaredTimes = [gState.scaredTimer for gState in currentGameState.getGhostStates()]
    capsulesList = currentGameState.getCapsules()
    scaredGhostList = [ghostList[i] for i in range(len(scaredTimes)) if scaredTimes[i] >= manhattanDistance(pos, ghostList[i]) * 2]
    bravedGhostList = [ghost for ghost in ghostList if ghost not in scaredGhostList]

    mazeDist = {}
    mazeDist[pos] = 0
    canEatDist = {}
    ghostDist = {}
    for food in foodList:
        canEatDist[food] = 1e9
    for capsule in capsulesList:
        canEatDist[capsule] = 1e9
    for ghost in ghostList:
        ghostDist[ghost] = 1e9

    queue = util.Queue()
    queue.push(pos)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while not queue.isEmpty():
        x, y = queue.pop()
        dist = mazeDist[(x, y)]
        if (x, y) in canEatDist:
            canEatDist[(x, y)] = dist
        elif (x, y) in ghostDist:
            ghostDist[(x, y)] = dist
        for dr in dirs:
            dx, dy = dr[0], dr[1]
            newx, newy = x + dx, y + dy
            if (newx, newy) in mazeDist or walls[newx][newy] or (newx, newy) in bravedGhostList:
                continue
            queue.push((newx, newy))
            mazeDist[(newx, newy)] = dist + 1
    
    inv_foodDist = [1.0 / canEatDist[food] for food in foodList]
    inv_capDist = [1.0 / canEatDist[capsule] for capsule in capsulesList]
    inv_scaredGPos = [1.0 / ghostDist[gPos] for gPos in scaredGhostList]
    inv_braveGPos = [1.0 / ghostDist[gPos] for gPos in bravedGhostList]
    score = scoreEvaluationFunction(currentGameState)
    if len(inv_foodDist) != 0:
        #score += max(inv_foodDist)
        score += sum(inv_foodDist)
    if len(inv_capDist) != 0:
        score += 100 * sum(inv_capDist)
    if len(scaredGhostList) != 0:
        #score += max(inv_scaredGPos)
        score += 200 * sum(inv_scaredGPos)
    if len(bravedGhostList) != 0:
        score += 1000 * sum(inv_braveGPos)
    return score

# Abbreviation
better = betterEvaluationFunction

