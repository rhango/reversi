from enum import IntEnum
from abc import ABCMeta, abstractmethod
from tail_recursive import *

class Disk(IntEnum):
    null  =  0
    dark  =  1
    light = -1

    def __str__(self):
        if self == Disk.dark:
            return "Dark"
        elif self == Disk.light:
            return "Light"
        else:
            return "Null"

    def reverse(self):
        return Disk(-1 * self)

class Result(IntEnum):
    win  =  1
    lose = -1
    draw =  0

    def __str__(self):
        if self == Result.win:
            return "Win"
        elif self == Result.lose:
            return "Lose"
        else:
            return "Draw"

class Player(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def setup_player(self, game, color):
        self.game  = game
        self.color = color
        return self

    @abstractmethod
    def tell_your_turn(self):
        pass

    @abstractmethod
    def tell_game_result(self, result):
        pass

class Board:
    _directions = {
        (x, y)
        for x in (-1, 0, 1)
            for y in (-1, 0, 1)
                if x != 0 or y != 0 }

    def __init__(self):
        self.array = [ [
                    Disk.null
                for col in range(9) ]
            for row in range(9) ]

        self.array[3][3:4] = [ Disk.light, Disk.dark  ]
        self.array[4][3:4] = [ Disk.dark,  Disk.light ]

        self.current_turn = Disk.dark
        self.update_possible_place()

    def _can_reverse_line(self, pos, dir):
        enemy = self.current_turn.reverse()
        p = [ pos[0]+dir[0], pos[1]+dir[1] ]

        if self.array[p[0]][p[1]] != enemy:
            return False

        while True:
            p = [ p[0]+dir[0], p[1]+dir[1] ]
            if self.array[p[0]][p[1]] == enemy:
                continue
            elif self.array[p[0]][p[1]] == self.current_turn:
                return True
            else:
                return False

    def _can_put_disk(self, row, col):
        if self.array[row][col] != Disk.null:
            return False

        return any([ self._can_reverse_line((row,col), dir) for dir in self._directions ])

    def update_possible_place(self):
        self.possible_place = [ [
                    self._can_put_disk(row, col)
                for col in range(8) ]
            for row in range(8) ]

    def exist_possible_place(self):
        return any([ any(possible_row) for possible_row in self.possible_place ])

    def _reverse_line(self, pos, dir):
        if self._can_reverse_line(pos, dir):
            enemy = self.current_turn.reverse()
            p = [ pos[0]+dir[0], pos[1]+dir[1] ]

            while self.array[p[0]][p[1]] == enemy:
                self.array[p[0]][p[1]] = self.array[p[0]][p[1]].reverse()
                p = [ p[0]+dir[0], p[1]+dir[1] ]

    def put_disk(self, row, col):
        self.array[row][col] = self.current_turn

        for dir in self._directions:
            self._reverse_line((row,col), dir)

    def check_game_result(self):
        s = sum([ sum(array_row[0:8]) for array_row in self.array[0:8] ])

        if s > 0:
            return { Disk.dark: Result.win, Disk.light: Result.lose }
        elif s < 0:
            return { Disk.dark: Result.lose, Disk.light: Result.win }
        else:
            return { Disk.dark: Result.draw, Disk.light: Result.draw }

class Reversi:
    def __init__(self, dark_player, light_player, render=None):
        self.player = {
            Disk.dark : dark_player.setup_player(  self, Disk.dark  ),
            Disk.light: light_player.setup_player( self, Disk.light ) }
        self.board = Board()
        self._prev_player_passed = False
        self.need_game_stop = False
        self._render = render

    def start_game(self):
        return self._process_turn()

    @TailRecursive()
    def tell_where_put(self, row, col):
        self.board.put_disk(row, col)
        self.board.current_turn = self.board.current_turn.reverse()
        return self._process_turn()

    @TailRecursive()
    def _process_turn(self):
        if self.need_game_stop:
            return None

        if self._render:
            self._render()

        self.board.update_possible_place()
        if not self.board.exist_possible_place():
            if self._prev_player_passed:
                result = self.board.check_game_result()
                for player, res in zip(self.player.values(), result.values()):
                    player.tell_game_result(res)
                return result
            else:
                self._prev_player_passed = True
                self.board.current_turn = self.board.current_turn.reverse()
                return self._process_turn()
        else:
            self._prev_player_passed = False
            return self.player[self.board.current_turn].tell_your_turn()
