import pickle
import numpy as np
import random

class QLearningAgent:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.load_q_table()

    def load_q_table(self):
        try:
            with open("data/test.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def save_q_table(self):
        with open("data/test.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def get_state(self, board):
        return "".join(board)

    def get_reward(self, game, num_moves):
        if game.current_winner == self.player:
            return 5
        elif game.current_winner == "O":
            return -5
        elif not game.empty_squares():
            return 0.5
        elif num_moves > 2:
            return -0.25
        else:
            return 0

    def choose_action(self, board, game):
        state = self.get_state(board)
        # print(f"state: {state} | q_table: {self.q_table[state]}")
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return random.choice([i for i in range(9) if board[i] == " "])
        else:
            q_values = [self.q_table[state][move] if move in self.q_table[state] else 0 for move in game.available_moves()]
            max_q_value = max(q_values)
            best_moves = [move for move, q_value in zip(game.available_moves(), q_values) if q_value == max_q_value]
            return random.choice(best_moves)

    def update_q_values(self, history, reward):
        for state, action in reversed(history):
            if state not in self.q_table:
                self.q_table[state] = np.zeros(9)
            self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][
                action
            ] + self.alpha * (reward + self.gamma * np.max(self.q_table[state]))
            reward *= self.gamma

    def make_action(self, game, history, player):
        state = self.get_state(game.board)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
    
        action = self.choose_action(game.board, game)
        game.make_move(action, player)
        history.append((state, action))
        reward = self.get_reward(game, len(history))
        self.update_q_values(history, reward)
        

    def train(self, episodes=1000):
        for _ in range(episodes):
            game = TicTacToe()
            history_player = []
            history_opponent = []
            turn = 'X'

            while game.empty_squares():
                if turn == self.player:
                    self.make_action(game, history_player, self.player)
                    turn = "O"
                else:
                    self.make_action(game, history_opponent, "O")
                    turn = self.player
        self.save_q_table()
