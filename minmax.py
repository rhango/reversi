from copy import deepcopy
import numpy as np
from engine import *
from players import *
from dqn import *

def eval_result(result):
    if result == Result.win:
        return 10
    elif result == Result.lose:
        return -10
    else:
        return 0

class MinMaxPlayer(Player):
    def __init__(self, depth, eval_func):
        self._depth = depth
        self._eval_func = eval_func

    def tell_your_turn(self):
        place, _ = self.minmax(self._depth, self.game.board)
        return self.game.tell_where_put(*place)

    def tell_game_result(self, result):
        pass

    def minmax(self, n, board):
        if n <= 0:
            return self._eval_func(self.color, board)
        else:
            vals = []
            possible_list = get_possible_place_list(board.possible_place)
            for place in possible_list:
                game = Reversi(DummyPlayer(), DummyPlayer(), board=deepcopy(board))
                game.start_game()
                result = game.tell_where_put(*place)

                if result is not None:
                    val = eval_result(result[self.color])
                else:
                    _, val = self.minmax(n-1, game.board)
                vals.append((place, val))

            if board.current_turn is self.color:
                return max(vals, key=lambda x: x[1])
            else:
                return min(vals, key=lambda x: x[1])

    @staticmethod
    def generate_player(depth, eval_func):
        def _gen_minmax_player():
            return MinMaxPlayer(depth, eval_func)
        return _gen_minmax_player

class AlphaBetaPlayer(Player):
    def __init__(self, depth, eval_func):
        self._depth = depth
        self._eval_func = eval_func

    def tell_your_turn(self):
        place, _ = self.alphabeta(self._depth, self.game.board)
        return self.game.tell_where_put(*place)

    def tell_game_result(self, result):
        pass

    def alphabeta(self, n, board, alpha=-np.inf, beta=np.inf):
        if n <= 0:
            return self._eval_func(self.color, board)
        else:
            action = None
            possible_list = get_possible_place_list(board.possible_place)
            for place in possible_list:
                game = Reversi(DummyPlayer(), DummyPlayer(), board=deepcopy(board))
                game.start_game()
                result = game.tell_where_put(*place)

                if result is not None:
                    val = eval_result(result[self.color])
                else:
                    _, val = self.alphabeta(n-1, game.board, alpha, beta)
                    if val is None:
                        continue

                if board.current_turn is self.color:
                    if val >= beta:
                        return None, None
                    elif val > alpha:
                        alpha  = val
                        action = place
                else:
                    if val <= alpha:
                        return None, None
                    elif val < beta:
                        beta   = val
                        action = place

            if board.current_turn is self.color:
                if action is None:
                    return None, -np.inf
                else:
                    return action, alpha
            else:
                if action is None:
                    return None, np.inf
                else:
                    return action, beta

    @staticmethod
    def generate_player(depth, eval_func):
        def _gen_alphabeta_player():
            return AlphaBetaPlayer(depth, eval_func)
        return _gen_alphabeta_player
