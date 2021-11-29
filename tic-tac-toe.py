from collections import deque
import numpy as np
import os
import time
import pickle

from numpy.random.mtrand import random

# dictionary containing states that won and which player won them so don't have to repeat computation for that.
board_state_evals = {}

state_index = {}

board_states_file = "board_states.p"

decay = 0.8

class TicTacToe():
    def __init__(self):
        self.board = [""]*9
        self.num_moves = 0

        self.symbols = ["X", "O"]

    def reset(self):
        self.board = [""]*9
        self.num_moves = 0

    def setState(self, state):
        self.num_moves = 0

        for i in range(len(state)):
            self.board[i] = state[i]

            if (state[i] != ""):
                self.num_moves += 1

    # returns which player's turn it is. Returns 0 or 1
    def playerTurn(self):
        return self.num_moves % 2

    def getAllValidMoves(self):
        if self.checkWin() != -1:
            return []
        
        moves = []
        for i in range(len(self.board)):
            if (self.board[i]!=""):
                continue

            moves.append(i)
        
        return moves

    # return boolean to indicate if move is allowed or not
    def makeMove(self, move, playerIndex):
        if (playerIndex < 0 or playerIndex > 1):
            return False

        if (self.board[move] == ""):
            self.board[move] = self.symbols[playerIndex]
            self.num_moves += 1
            return True
        
        return False
    
    # returns which player won. Returns -1 if no player has won yet. Return 0 or 1 for winning player
    def checkWin(self):
        boardState = tuple(self.board)
        if (boardState in board_state_evals):
            return board_state_evals[boardState]

        # check rows
        for i in range(3):
            symbol = self.board[i*3]
            
            if symbol == "":
                continue
            
            won = True
            for j in range(1, 3):
                index = i*3 + j

                if (self.board[index] != symbol):
                    won = False
                    break

            if won:
                winner = self.symbols.index(symbol)

                board_state_evals[boardState] = winner

                return winner

        # check columns
        for i in range(3):
            symbol = self.board[i]
            
            if symbol == "":
                continue
            
            won = True
            for j in range(1, 3):
                index = i + j*3

                if (self.board[index] != symbol):
                    won = False
                    break

            if won:
                winner = self.symbols.index(symbol)

                board_state_evals[boardState] = winner

                return winner

        # check diagonals
        diagonals = [(0,4,8), (2,4,6)]

        for diag in diagonals:
            symbol = self.board[diag[0]]
            
            if symbol == "":
                continue
            
            won = True

            for j in range(1, 3):
                index = diag[j]
                if (self.board[index] != symbol):
                    won = False
                    break
            
            if won:
                winner = self.symbols.index(symbol)

                board_state_evals[boardState] = winner

                return winner

        board_state_evals[boardState] = -1
        return -1

    def get_state(self):
        return tuple(self.board)
                
def generate_all_states():
    board = TicTacToe()
    board.reset()

    states = set()

    queue = deque()

    queue.append(board.get_state())

    while(len(queue)>0):
        state = queue.popleft()
        states.add(state)

        board.setState(state)

        
        moves = board.getAllValidMoves()
        for move in moves:
            board.setState(state)
            board.makeMove(move, board.playerTurn())

            queue.append(board.get_state())
    
    return list(states)

def get_policy_from_values(states, V):
    policy = {}

    board = TicTacToe()
    board.reset()

    for state in states:
        q_values = []
            
        board.setState(state)

        actions = board.getAllValidMoves()

        for a in actions:
            board.setState(state)
            playerTurn = board.playerTurn()
            board.makeMove(a, playerTurn)

            win = board.checkWin()

            reward = 0
            if (win == 0):
                reward = 1
            elif(win == 1):
                reward = -1

            next_state = board.get_state()

            value = reward + decay*V[next_state]
            q_values.append(value)

        if len(q_values) == 0:
            continue

        board.setState(state)
        playerTurn = board.playerTurn()

        index = 0
        if (playerTurn==0):
            index = np.argmax(q_values)
        elif (playerTurn==1):
            index = np.argmin(q_values)

        policy[state] = actions[index]
    
    return policy

def get_values_from_policy(states, policy):
    A = np.eye(len(states))
    b = np.zeros(len(states))

    board = TicTacToe()

    for i in range(len(states)):
        if (states[i] not in policy):
            continue

        action = policy[states[i]]

        board.setState(states[i])

        board.makeMove(action, board.playerTurn())

        win = board.checkWin()

        reward = 0
        if (win == 0):
            reward = 1
        elif(win == 1):
            reward = -1
        
        next_state = board.get_state()
        j = state_index[next_state]

        A[i][j] -= decay

        b[i] += reward
    
    V = np.linalg.solve(A, b)
    values = {}

    for i in range(len(states)):
        values[states[i]] = V[i]

    return values

def value_iteration(states):
    V = {}
    for state in states:
        V[state] = np.random.uniform(-0.5, 0.5)

    done = False

    board = TicTacToe()
    board.reset()

    for iter in range(100):
        done = True
        for state in states:
            q_values = []
            
            board.setState(state)

            actions = board.getAllValidMoves()

            for a in actions:
                board.setState(state)
                playerTurn = board.playerTurn()
                board.makeMove(a, playerTurn)

                win = board.checkWin()

                reward = 0
                if (win == 0):
                    reward = 1
                elif(win == 1):
                    reward = -1

                next_state = board.get_state()

                value = reward + decay*V[next_state]
                q_values.append(value)
            
            if len(q_values) == 0:
                continue

            board.setState(state)
            playerTurn = board.playerTurn()

            best = 0
            if (playerTurn==0):
                best = max(q_values)
            elif (playerTurn==1):
                best = min(q_values)

            if (abs(V[state]-best)>0.0001):
                done = False
            
            V[state] = best

        if (done):
            print(iter, " iterations required to converge")
            break
    
    return V

def policy_iteration(states):
    policy = {}

    board = TicTacToe()

    for state in states:
        board.setState(state)
        actions = board.getAllValidMoves()

        if (len(actions) == 0):
            continue

        policy[state] = np.random.choice(actions)
    
    for i in range(100):
        V = get_values_from_policy(states, policy)
        new_policy = get_policy_from_values(states, V)

        done = True

        for state in new_policy.keys():
            if (new_policy[state]!=policy[state]):
                policy[state] = new_policy[state]
                done = False

        if (done):
            print(i, " iterations required to converge")
            break

    return policy

def q_learning(states):
    q_values = {}

    board = TicTacToe()
    for state in states:
        q_values[state] = {}
        board.setState(state)
        actions = board.getAllValidMoves()

        for action in actions:
            q_values[state][action] = np.random.uniform(-0.5, 0.5)

    epsilon = 0.7
    min_epsilon = 0.15
    epsilon_decay = 0.0001
    visits = {}

    for i in range(100000):
        done = False
        board.reset()

        while not done:
            actions = board.getAllValidMoves()

            if (len(actions)==0):
                done = True
                break

            action = np.random.choice(actions)

            if (np.random.random() > epsilon):
                action_values = q_values[board.get_state()]

                best_action = None
                best_q_value = 0
                for action_key in action_values.keys():
                    if (best_action == None):
                        best_action = action_key
                        best_q_value = action_values[action_key]
                    elif (action_values[action_key] > best_q_value):
                        best_action = action_key
                        best_q_value = action_values[action_key]
                
                if (best_action != None):
                    action = best_action
            
            learn_rate = 0.5

            last_state = board.get_state()

            board.makeMove(action, board.playerTurn())

            win = board.checkWin()

            reward = 0
            if (win == 0):
                reward = 1
            elif(win == 1):
                reward = -1

            playerTurn = board.playerTurn()

            best = 0

            if (len(q_values[board.get_state()])>0):
                if (playerTurn==0):
                    best = q_values[board.get_state()][get_key_with_max_value(q_values[board.get_state()])]
                elif (playerTurn==1):
                    best = q_values[board.get_state()][get_key_with_min_value(q_values[board.get_state()])]

            q_values[last_state][action] = (1-learn_rate)*q_values[last_state][action] + learn_rate*(reward + decay * best)
        
        epsilon = max(min_epsilon, 0.7*np.exp(-epsilon_decay*i))

    return q_values


def play_policy(policy):
    board = TicTacToe()

    humanIndex = np.random.randint(2)
    robotIndex = -1

    if (humanIndex == 0):
        robotIndex = 1
    else:
        robotIndex = 0

    done = False
    turns = 0

    while not done:
        if (turns % 2 == humanIndex):
            move = int(input("Input move here: "))
            while not board.makeMove(move, humanIndex):
                move = int(input("Input move here: "))
            
            display_board(board.get_state())

            winner = board.checkWin()
            if (winner != -1):
                done = True
            elif(board.num_moves==9):
                done = True
        else:
            move = policy[board.get_state()]
            board.makeMove(move, robotIndex)

            display_board(board.get_state())

            winner = board.checkWin()
            if (winner != -1):
                done = True
            elif(board.num_moves==9):
                done = True

        turns += 1

def play_policy_Q(q_table):
    board = TicTacToe()

    humanIndex = np.random.randint(2)
    robotIndex = -1

    if (humanIndex == 0):
        robotIndex = 1
    else:
        robotIndex = 0

    done = False
    turns = 0

    while not done:
        if (turns % 2 == humanIndex):
            move = int(input("Input move here: "))
            while not board.makeMove(move, humanIndex):
                move = int(input("Input move here: "))
            
            display_board(board.get_state())

            winner = board.checkWin()
            if (winner != -1):
                done = True
            elif(board.num_moves==9):
                done = True
        else:
            move = None

            if (robotIndex == 0):
                move = get_key_with_max_value(q_table[board.get_state()])
            elif (robotIndex == 1):
                move = get_key_with_min_value(q_table[board.get_state()])

            board.makeMove(move, robotIndex)

            display_board(board.get_state())

            winner = board.checkWin()
            if (winner != -1):
                done = True
            elif(board.num_moves==9):
                done = True

        turns += 1

def get_key_with_max_value(dictionary):
    highest_val = 0
    highest_key = None

    for k, v in dictionary.items():
        if (highest_key == None):
            highest_key = k
            highest_val = v
        elif (v > highest_val):
            highest_val = v
            highest_key = k

    return highest_key

def get_key_with_min_value(dictionary):
    lowest_val = 0
    lowest_key = None
    
    for k, v in dictionary.items():
        if (lowest_key == None):
            lowest_key = k
            lowest_val = v
        elif (v < lowest_val):
            lowest_val = v
            lowest_key = k

    return lowest_key


def display_board(board):
    for i in range(3):
        line = ""
        for j in range(3):
            index = i*3 + j
            piece = board[index]
            if piece == "":
                piece = "_"
            
            line += piece + "\t"

        print(line)
    print()

if (__name__ == "__main__"):
    states = []
    if os.path.exists(board_states_file):
        states = pickle.load(open(board_states_file, "rb"))
    else:
        states = generate_all_states()
        pickle.dump(states, open(board_states_file, "wb"))

    for i in range(len(states)):
        state_index[states[i]] = i
    
    #start = time.time()
    #values = value_iteration(states)
    #total = time.time() - start
    #print(total)
    #policy = get_policy_from_values(states, values)
    
    #start = time.time()
    #policy = policy_iteration(states)
    #total = time.time() - start
    #print(total)

    #play_policy(policy)

    q_table = q_learning(states)

    for i in range(5):
        play_policy_Q(q_table)