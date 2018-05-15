from copy import deepcopy
import numpy as np
from engine import *
from players import *
from dqn import *

class MinMaxPlayer(Player):
    def __init__(self, eval_func):
        self._eval_func = eval_func

    def tell_your_turn(self):
        place, _ = self.minmax(3, self.game.board)
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
                    res = result[self.color]
                    if res == Result.win:
                        val = 10
                    elif res == Result.lose:
                        val = -10
                    else:
                        val = 0
                else:
                    _, val = self.minmax(n-1, game.board)
                vals.append((place, val))

            if board.current_turn is self.color:
                return max(vals, key=lambda x: x[1])
            else:
                return min(vals, key=lambda x: x[1])

    @staticmethod
    def generate_player(eval_func):
        def _gen_minmax_player():
            return MinMaxPlayer(eval_func)
        return _gen_minmax_player
