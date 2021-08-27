# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from sys import path
import util
import searchAgents
from math import atan2,degrees
from game import Directions
from game import Actions
import sys
sys.setrecursionlimit(5000)
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    startState = problem.getStartState()
    startNode = startState, '', 0, [] 
    queue.push(startNode)
    closed = []
    while queue :
        outNode = queue.pop()
        state, action, cost, path = outNode
        if state not in closed :
            closed.append(state)
            
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break
            succ = problem.getSuccessors(state)
            for successor in succ:
                stateSucc, actionSucc, costSucc = successor
                nextNode = (stateSucc, actionSucc, cost + costSucc, path + [(state, action)])
                queue.push(nextNode)
    operations=[]
    for operation in path:
        operations.append(operation[1])
    del operations[0]
    return operations


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 2 ***"
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = startState, '', 0, [] 
    closed = []
    operations = []
    hStart = heuristic(startState, problem)
    priorityQueue.push(startNode,hStart)
    bestG = hStart + 0
    isGoal = False
    while not priorityQueue.isEmpty():
        outNode = priorityQueue.pop()
        outState, outAction, outCost, outPath = outNode
        
        if (outState not in closed) or (outCost < bestG): 
            closed.append(outState)
            bestG = outCost
            if problem.isGoalState(outState):
                outPath = outPath + [(outState,outAction)]
                isGoal = True
                break

            succ = problem.getSuccessors(outState)
            for successor in succ:
                stateSucc, actionSucc, costSucc = successor
                nextNode = (stateSucc, actionSucc, costSucc + outCost, outPath + [(outState,outAction)])
                h = costSucc + outCost+heuristic(stateSucc, problem)
                if h < float('inf') :
                    priorityQueue.push(nextNode,h)
    if isGoal == True:
        for operation in outPath:
            operations.append(operation[1])
        del operations[0]
    else:
        operations = []
    return operations


# Extensions Assignment 1
def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 1 ***"
    initialNode = (problem.getStartState(),'',0,[])
    operations = []
    while not problem.isGoalState(initialNode[0]):
        nextNode = improve(problem,initialNode,heuristic)
        initialNode = nextNode
    if problem.isGoalState(initialNode[0]):
        state,action,cost,path = initialNode
        path = path + [(state,action)]
        for operation in path:
            operations.append(operation[1])
        
    del operations[0]
    return operations 
    
def improve(problem, initialNode,heuristic):
    queue = util.Queue()
    state0 = initialNode[0]
    queue.push(initialNode)
    closed = set()
    
    while not queue.isEmpty():
        outNode = queue.pop()
        state, action, cost, path = outNode
        
        if state not in closed:
            closed.add(state)
            if heuristic(state,problem) < heuristic(state0,problem):
                return outNode
                
            succ = problem.getSuccessors(state)
            for successor in succ:
                stateSucc, actionSucc, costSucc = successor
                nextNode = (stateSucc, actionSucc, costSucc+cost, path+[(state, action)])
                queue.push(nextNode)
    return outNode


def jumpPointSearch1(problem, heuristic=nullHeuristic):
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = startState, '', 0, [] 

    
    operations = []
    fStart = heuristic(startState, problem)
    priorityQueue.push(startNode,fStart)
    bestG = fStart
    isGoal = False
    closed = []
    while not priorityQueue.isEmpty():
        currentNode = priorityQueue.pop()
        # currentNode is composed of (currentState/coord, currentAction, currentCost, currentPath)
        currentState, currentAction, currentCost, currentPath = currentNode
        if currentState not in closed:
            closed.append(currentState)
        if problem.isGoalState(currentState):
            isGoal = True
            break
        succ = identifySuccessors((currentState, currentAction, currentCost), problem, closed)

        #print(closed)
        print(succ)
        for successor in succ:
            if successor is not None:

                succState, succAction, succCost = successor

                tempCost = util.manhattanDistance(succState, currentState) + succCost
                #costSucc = util.manhattanDistance(stateSucc, outState) + costSucc

                fNext = tempCost + heuristic(succState, problem)
                nextNode = (succState, succAction, succCost, currentPath + [(currentState, currentAction)])
                #nextNode = (stateSucc, actionSucc, costSucc, outPath + [(outState, outAction)])
                priorityQueue.push(nextNode, fNext)
                #openset.append(stateSucc)

    if isGoal == True:
        for operation in currentPath:
            operations.append(operation[1])
        del operations[0]
    else:
        operations = []
  
    return operations

def jumpPointSearch(problem, heuristic=nullHeuristic):
    """
    Search by pruning non-critical neighbors with jump points.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 3 ***"
    priorityQueue = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = startState, '', 0, []

    closed = []
    operations = []
    fStart = heuristic(startState, problem)
    priorityQueue.push(startNode, fStart)
    isGoal = False

    while not priorityQueue.isEmpty():
        outNode = priorityQueue.pop()
        outState, outAction, outCost, outPath = outNode
        if outState not in closed:
            closed.append(outState)

            if problem.isGoalState(outState):
                #outPath = outPath + [(outState, outAction)]
                isGoal = True
                print(outPath)
                break
            succ = identifySuccessors(outNode, problem, closed)
            print(succ)
            for successor in succ:
            #if successor != None:
                stateSucc, actionSucc, costSucc, _ = successor

                costSucc = util.manhattanDistance(stateSucc, outState) + costSucc
                fNext = costSucc + heuristic(stateSucc, problem)

                #nextNode = (stateSucc, actionSucc, costSucc, outPath + [(outState, outAction)])
                priorityQueue.push(successor, fNext)


    if isGoal == True:
        for operation in outPath:
            operations.append(operation[1])

    else:
        operations = []


    return operations

def identifySuccessors(currentNode,problem,closed):
    successor = []

    currentState, _, _, _ = currentNode
    neighbors = problem.getSuccessors(currentState)

    for n in neighbors:
        stateN = n[0]
        if stateN not in closed:

            dir = direction(currentState, n[0])
            retN = jump(currentNode, dir, problem)
            successor.append(retN)

    successor = [x for x in successor if x is not None]
    print(successor)
    return successor
   
    
        
def jump(startNode,dir,problem):
    
    parentPos = startNode[0] #(7,5)
    vector, action, isVertical = dir
    n = move(startNode,action)
    # n is : (coord, Action, Coat)
    statePos, startAction, cost, _ = n

    (nodeX, nodeY) = statePos
     
    wallGrid= problem.walls
    print(wallGrid, '\n')
    widthWall = wallGrid.width - 2 # 7
    heightWall = wallGrid.height-2 # 5
    
    if problem.walls[nodeX][nodeY] or (nodeX >widthWall) or (nodeY> heightWall) or nodeX<1 or nodeY<1:
        return None
    
    if problem.isGoalState(statePos):
        return n
    
    neighbours = problem.getSuccessors(statePos)
   
    # if node is forced
    parentX = parentPos[0]
    parentY = parentPos[1]

    print(action, Directions)
    for neighbour in neighbours:
        neighbourPos = neighbour[0]

        if neighbour[0] != parentPos and (not problem.walls[neighbourPos[0]][neighbourPos[1]]):


            if action == Directions.EAST:
                upWall = problem.walls[parentX][parentY+1]
                downWall = problem.walls[parentX][parentY-1]
                if (upWall and not problem.walls[parentX+1][parentY+1]) or (downWall and not problem.walls[parentX+1][parentY-1]):
                    return n

            if action == Directions.WEST:
                upWall = problem.walls[parentX][parentY+1]
                downWall = problem.walls[parentX][parentY-1]

                if (upWall and not problem.walls[parentX-1][parentY-1]) or (downWall and not problem.walls[parentX+1][parentY-1]):
                    return n

            if action == Directions.SOUTH:
                rightWall = problem.walls[parentX+1][parentY]
                leftWall = problem.walls[parentX-1][parentY]
                if (rightWall and not problem.walls[parentX-1][parentY+1]) or (leftWall and not problem.walls[parentX+1][parentY-1]):
                    return n

            if action == Directions.NORTH:
                rightWall = problem.walls[parentX+1][parentY]
                leftWall = problem.walls[parentX-1][parentY]
                if (rightWall and not problem.walls[parentX-1][parentY+1])  or (leftWall and not problem.walls[parentX-1][parentY-1]):
                    return n

    return jump(n,dir,problem)


def valid_position(width, height, x, y):
    return 0 < x <= width and 0 < y <= height



def reverseDirection(direction):
    d, action,isVertical = direction
    dNew = (-d[0],-d[1])
    if action == Directions.NORTH:
        action = Directions.SOUTH
    elif action == Directions.SOUTH:
        action = Directions.NORTH
    elif action == Directions.WEST:
        action = Directions.EAST
    elif action == Directions.EAST:
        action = Directions.WEST
    newDirection = dNew,action,isVertical
    return newDirection
        


def move(node,action):
    startPos, dStart, startCost, lastPath = node
    dx, dy = Actions.directionToVector(action)
    
    nodeX,nodeY = int(startPos[0] + dx), int(startPos[1] + dy)
    node2 = (nodeX,nodeY)

    n = node2, action, startCost + 1, lastPath+[(startPos, action)]
    return n
   

'''
Return vector from pointB to pointA, eg.(-1.0 0.0)
'''
def direction(nodeA,nodeB):
    x = nodeB[0] - nodeA[0]
    y = nodeB[1] - nodeA[1]
    angle = degrees(atan2(y,x))
    isVertical = True
    
    action = Directions.NORTH
    direct = Actions.directionToVector(action)
    if angle == 90.0:
        action = Directions.NORTH
        direct = Actions.directionToVector(action)
    elif angle == -90.0:
        action = Directions.SOUTH
        direct = Actions.directionToVector(action)
    elif angle == 180.0:
        action = Directions.WEST
        direct = Actions.directionToVector(action)
        isVertical = False
    elif angle == 0.0:
        action = Directions.EAST
        direct = Actions.directionToVector(action)
        isVertical = False
    return (direct, action, isVertical) 





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ehc = enforcedHillClimbing
jps = jumpPointSearch
