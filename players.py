import time
import random
from engine import *

def delay(func):
    def _func():
        player = func()
        player.tell_your_turn = _delay_func(player.tell_your_turn)
        return player
    return _func

def _delay_func(func):
    def _func():
        time.sleep(0.5)
        return func()
    return _func

class Human(Player):
    def __init__(self, tell_win_or_lose):
        self._tell_win_or_lose = tell_win_or_lose
        self._my_turn = False

    def tell_your_turn(self):
        self._my_turn = True
        return None

    def tell_where_clicked(self, row, col):
        if self._my_turn:
            if self.game.board.possible_place[row][col]:
                self._my_turn = False
                self.game.tell_where_put(row, col)

    def tell_game_result(self, result):
        self._tell_win_or_lose(self.color, result)

class Random(Player):
    def tell_your_turn(self):
        place_list = [
            (row, col)
            for row, possible_row in enumerate(self.game.board.possible_place)
                for col, possible in enumerate(possible_row)
                    if possible ]

        place = random.choice(place_list)
        return self.game.tell_where_put(*place)

    def tell_game_result(self, result):
        pass
