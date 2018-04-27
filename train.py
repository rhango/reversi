import sys
import csv
import time
from engine import *
from players import *
from dqn import *

class Timer:
    def __init__(self):
        pass

    def start(self):
        self._start_time = time.time()

    def get_time(self):
        return time.time() - self._start_time

def non_delay(player_gen):
    def _non_delay_player_gen():
        return player_gen(delay=0)
    return _non_delay_player_gen

class Battle:
    def __init__(self, player_generators, n_games):
        self._player_gen = player_generators
        self._n_games    = n_games

    def func_before_battle(self):
        pass

    def func_before_game(self, game_idx):
        pass

    def __call__(self):
        self.func_before_battle()

        for game_idx in range(1, self._n_games + 1):
            self.func_before_game(game_idx)

            game = Reversi(
                self._player_gen[ (game_idx + 1) % 2 ](),
                self._player_gen[  game_idx      % 2 ]() )
            result = game.start_game()

            self.func_after_game(game_idx, result)

        self.func_after_battle()

    def func_after_game(self, game_idx, result):
        pass

    def func_after_battle(self):
        pass

class Test(Battle):
    fieldnames = [ "N Games", "Dark WR", "Light WR", "Total WR", "Time" ]

    def __init__(self, ai, tester_gen, n_tests, timer=None, file_name=None):
        super().__init__( (non_delay(ai.generate_player), non_delay(tester_gen)), n_tests )
        self._n_tests = n_tests
        self._timer   = timer

        self._data = { fieldname: None for fieldname in self.fieldnames }

        if file_name is not None:
            self._need_export = True
            self._file   = open('data/{}.csv'.format(file_name), 'w')
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            self._need_export = False

    def func_before_battle(self):
        self._wr = { Disk.dark: 0, Disk.light: 0 }

    def __call__(self, game_idx=None):
        self._game_idx = game_idx
        super().__call__()
        return self._data

    def func_after_game(self, test_idx, result):
        if (test_idx + 1) % 2 == 0:
            ai_color = Disk.dark
        else:
            ai_color = Disk.light

        if result[ai_color] is Result.win:
            self._wr[ai_color] += 1

    def func_after_battle(self):
        if self._game_idx is not None:
            self._data["N Games"] = self._game_idx
        else:
            self._data["N Games"] = "--"
        for color in (Disk.dark, Disk.light):
            self._data[ "{} WR".format(str(color)) ] = self._wr[color] / (self._n_tests/2)
        self._data["Total WR"] = sum(self._wr.values()) / self._n_tests
        if self._timer is not None:
            self._data["Time"] = self._timer.get_time()
        else:
            self._data["Time"] = "--"

        if self._need_export:
            self._writer.writerow(self._data)

    def __del__(self):
        if self._need_export:
            self._file.close()

class Train(Battle):
    def __init__(self, ai, ai_name, enemy_gen, n_games, save_timings, tester_gen, n_tests, test_interval):
        super().__init__( (ai.generate_trainer, enemy_gen), n_games )
        self._ai      = ai
        self._ai_name = ai_name
        self._save_timings = save_timings

        self._timer = Timer()
        self._test  = Test(self._ai, tester_gen, n_tests, self._timer, self._ai_name)
        self._test_interval = test_interval

        self._outctl = OutputController()

    def func_before_battle(self):
        self._timer.start()
        result = self._test(game_idx=0)
        self._outctl.output_test_result(result)

    def func_after_game(self, game_idx, result):
        self._outctl.update_game_idx(game_idx)
        if game_idx % self._test_interval == 0:
            result = self._test(game_idx)
            self._outctl.output_test_result(result)

        if game_idx in self._save_timings:
            agent_name = '{ai_name}-{game_idx}'.format(
                ai_name  = self._ai_name,
                game_idx = game_idx )
            self._ai.agent.save( 'DQN/{}'.format(agent_name) )

class OutputController:
    def __init__(self):
        self._idx_updating = False

        sys.stderr.write("\n")
        sys.stderr.flush()
        print("    ".join(("{:<9}".format(label) for label in Test.fieldnames)))

    def update_game_idx(self, game_idx):
        if not self._idx_updating:
            self._idx_updating = True
            sys.stderr.write("\n")

        sys.stderr.write("\rNum of games: {:>7}".format(game_idx))
        sys.stderr.flush()

    def output_test_result(self, result):
        if self._idx_updating:
            self._idx_updating = False
            sys.stderr.write("\033[1M\033[F")
            sys.stderr.flush()

        time = int(result["Time"])
        time, second = divmod(time, 60)
        hour, minute = divmod(time, 60)

        print("{n_games:>9}    {dark:>9.2%}    {light:>9.2%}    {total:>9.2%}    {time:9}".format(
            n_games = result["N Games"],
            dark    = result["Dark WR"],
            light   = result["Light WR"],
            total   = result["Total WR"],
            time    = "{:>3}:{:0>2}:{:0>2}".format(hour, minute, second)
        ))

def generate_ai(activation_func, n_layers, n_hidden_channels, decay_steps=50000, enemy='RND', gpu=0):
    ai = ReversiAI(activation_func, n_layers, n_hidden_channels, decay_steps=decay_steps, gpu=gpu)

    if enemy == 'RND':
        enemy_gen = non_delay(Random)
    elif enemy == 'DQN':
        enemy_ai = ReversiAI(activation_func, n_layers, n_hidden_channels, decay_steps=decay_steps, gpu=gpu)
        enemy_gen = enemy_ai.generate_trainer

    ai_name = '{activation_func}-{n_layers}x{n_hidden_channels}'.format(
        activation_func   = activation_func,
        n_layers          = n_layers,
        n_hidden_channels = n_hidden_channels )

    if decay_steps != 50000:
        ai_name += '-d' + decay_steps

    ai_name += '-vs' + enemy

    return ai, ai_name, enemy_gen

def main():
    ai, ai_name, enemy_gen = generate_ai(
        activation_func   = 'leaky_relu',
        n_layers          = 5,
        n_hidden_channels = 256,
        enemy             = 'RND' )

    ai_name += '-1'

    train = Train(
        ai            = ai,
        ai_name       = ai_name,
        enemy_gen     = enemy_gen,
        n_games       = 100000,
        save_timings  = {10000, 20000, 50000, 100000},
        tester_gen    = Random,
        n_tests       = 400,
        test_interval = 1000 )

    train()

if __name__ == '__main__':
    main()
