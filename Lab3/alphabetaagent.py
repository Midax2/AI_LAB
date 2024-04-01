from exceptions import AgentException

import math


class AlphaBetaAgent:
    def __init__(self, token):
        self.my_token = token

    def heuristic(self, connect4):
        corners = [(0, 0), (0, connect4.width - 1), (connect4.height - 1, 0), (connect4.height - 1, connect4.width - 1)]
        count = sum(1 for row, col in corners if connect4.board[row][col] == self.my_token)
        return 0.2 * count

    def alphabeta(self, connect4, x, d, alpha, beta):
        if self.my_token == connect4.wins:
            return 1
        if self.my_token != connect4.wins and connect4.wins is not None:
            return -1
        if connect4.wins is None and connect4.game_over is True:
            return 0
        if d == 0:
            return self.heuristic(connect4)
        temp_alpha = alpha
        temp_beta = beta
        if x == 1:
            best_move = -math.inf
            for i in range(len(connect4.possible_drops())):
                n_column = connect4.possible_drops()[i]
                connect4.drop_token(n_column)
                res = self.alphabeta(connect4, 0, d - 1, temp_alpha, temp_beta)
                connect4.undo_last_move()
                best_move = max(res, best_move)
                temp_alpha = max(temp_alpha, best_move)
                if best_move >= temp_beta:
                    break
            return best_move
        if x == 0:
            best_move = math.inf
            for i in range(len(connect4.possible_drops())):
                n_column = connect4.possible_drops()[i]
                connect4.drop_token(n_column)
                res = self.alphabeta(connect4, 1, d - 1, temp_alpha, temp_beta)
                connect4.undo_last_move()
                best_move = min(res, best_move)
                temp_beta = min(temp_beta, best_move)
                if best_move <= temp_alpha:
                    break
            return best_move

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        x = 1
        d = 4
        best_move = -math.inf
        alpha = -math.inf
        best_state = 0
        for i in range(len(connect4.possible_drops())):
            n_column = connect4.possible_drops()[i]
            connect4.drop_token(n_column)
            res = self.alphabeta(connect4, x, d - 1, alpha, math.inf)
            connect4.undo_last_move()
            if res > best_move:
                best_move = res
                best_state = i
        return connect4.possible_drops()[best_state]
