import copy, sys
from exceptions import AgentException


def basic_static_eval(connect4, player="o"):
    if player == connect4.wins:
        return float('inf')
    elif player != connect4.wins and connect4.wins is not None:
        return -float('inf')
    elif connect4.wins is None and connect4.game_over is True:
        return 0
    else:
        count_our = 0
        count_enemy = 0
        enemy = 'o' if player == 'x' else 'o'
        for four in connect4.iter_fours():
            if four.count(player) == 3:
                count_our += 1
            elif four.count(enemy) == 3:
                count_enemy += 1
        return count_our - count_enemy


def advanced_static_eval(connect4, player="o"):
    if player == connect4.wins:
        return float('inf')
    elif player != connect4.wins and connect4.wins is not None:
        return -float('inf')
    elif connect4.wins is None and connect4.game_over is True:
        return 0
    else:
        count_our = 0
        count_enemy = 0
        center_column = connect4.width // 2
        enemy = 'o' if player == 'x' else 'o'
        for four in connect4.iter_fours():
            if four.count(player) == 3:
                count_our += 1
            elif four.count(enemy) == 3:
                count_enemy += 1
        return sum(connect4.board[row][center_column] == player
                   for row in range(connect4.height)) + count_our - count_enemy



class AlphaBetaAgent:

    def __init__(self, my_token="o", heuristic_func=basic_static_eval):
        self.my_token = my_token
        self.heuristic_func = heuristic_func

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")

        best_move, best_score = self.alphabeta(connect4)
        return best_move

    def alphabeta(self, connect4, depth=4, maximizing=True, alpha=-float('inf'), beta=float('inf')):
        if connect4.game_over is True or depth == 0:
            best_score = self.heuristic_func(connect4, connect4.who_moves)
            best_move = None
            return best_move, best_score
        temp_alpha = alpha
        temp_beta = beta
        if maximizing is True:
            best_score = -float('inf')
            best_state = 0
            for i in range(len(connect4.possible_drops())):
                copy_connect = copy.deepcopy(connect4)
                n_column = connect4.possible_drops()[i]
                copy_connect.drop_token(n_column)
                move_now, score_now = self.alphabeta(copy_connect, depth - 1, False, temp_alpha, temp_beta)
                if score_now > best_score:
                    best_score = score_now
                    best_state = i
                temp_alpha = max(temp_alpha, best_score)
                if best_score >= temp_beta:
                    break
            best_move = connect4.possible_drops()[best_state]
            return best_move, best_score
        if maximizing is False:
            best_score = float('inf')
            best_state = 0
            for i in range(len(connect4.possible_drops())):
                copy_connect = copy.deepcopy(connect4)
                n_column = connect4.possible_drops()[i]
                copy_connect.drop_token(n_column)
                move_now, score_now = self.alphabeta(copy_connect, depth - 1, True, temp_alpha, temp_beta)
                if score_now < best_score:
                    best_score = score_now
                    best_state = i
                temp_beta = min(temp_beta, best_score)
                if best_score <= temp_alpha:
                    break
            best_move = connect4.possible_drops()[best_state]
            return best_move, best_score
