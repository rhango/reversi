import csv
import time
from reversi_engine import *
from reversi_players import *
from reversi_dqn import *

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
    _fieldnames = [ "N Games", "Dark WR", "Light WR", "Total WR", "Time" ]

    def __init__(self, ai, tester_gen, n_tests, timer=None, file_name=None):
        super().__init__( (non_delay(ai.generate_player), tester_gen), n_tests )
        self._n_tests = n_tests
        self._timer   = timer

        if file_name is not None:
            self._need_export = True
            self._file   = open('data/{}.csv'.format(file_name), 'w')
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            self._need_export = False

    def func_before_battle(self):
        self._wr = { Disk.dark: 0, Disk.light: 0 }

    def __call__(self, game_idx=None):
        self._game_idx = game_idx
        super().__call__()

    def func_after_game(self, test_idx, result):
        if (test_idx + 1) % 2 == 0:
            ai_color = Disk.dark
        else:
            ai_color = Disk.light

        if result[ai_color] is Result.win:
            self._wr[ai_color] += 1

    def func_after_battle(self):
        data = {}
        if self._game_idx is not None:
            data["N Games"] = self._game_idx
        else:
            data["N Games"] = "--"
        for color in (Disk.dark, Disk.light):
            data[ "{} WR".format(str(color)) ] = self._wr[color] / (self._n_tests/2)
        data["Total WR"] = sum(self._wr.values()) / self._n_tests
        if self._timer is not None:
            data["Time"] = self._timer.get_time()
        else:
            data["Time"] = "--"

        if self._need_export:
            self._writer.writerow(data)
        for fieldname in self._fieldnames:
            print("{0}: {1}; ".format(fieldname, data[fieldname]), end="")
        print()

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

    def func_before_battle(self):
        self._timer.start()
        #self._test(game_idx=0)

    def func_after_game(self, game_idx, result):
        print(game_idx)
        if game_idx % self._test_interval == 0:
            self._test(game_idx)

        if game_idx in self._save_timings:
            agent_name = '{ai_name}-{game_idx}'.format(
                ai_name  = self._ai_name,
                game_idx = game_idx )
            self._ai.agent.save( 'DQN/{}'.format(agent_name) )

    def func_after_battle(self):
        print("Total Time:", self._timer.get_time())

def create_ai(activation_func, n_layers, n_hidden_channels, decay_steps=50000, gpu=0):
    ai = ReversiAI(activation_func, n_hidden_channels, n_hidden_channels, decay_steps=decay_steps, gpu=gpu)

    ai_name = '{activation_func}-{n_layers}x{n_hidden_channels}-d{decay_steps}'.format(
        activation_func   = activation_func,
        n_layers          = n_layers,
        n_hidden_channels = n_hidden_channels,
        decay_steps       = decay_steps )

    return ai, ai_name

def main():
    ai, ai_name = create_ai(
        activation_func   = 'leaky_relu',
        n_layers          = 5,
        n_hidden_channels = 128,
        decay_steps       = 50000 )

    enemy_ai = ReversiAI('leaky_relu', 5, 128, gpu=0)
    ai_name += '-vsAI'

    train = Train(
        ai            = ai,
        ai_name       = ai_name,
        enemy_gen     = enemy_ai.generate_trainer,
        n_games       = 30000,
        save_timings  = {10000, 20000, 50000},
        tester_gen    = non_delay(Random),
        n_tests       = 100,
        test_interval = 1000 )

    train()

if __name__ == '__main__':
    main()
