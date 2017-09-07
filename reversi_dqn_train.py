import csv
import time
from reversi_engine import *
from reversi_players import *
from reversi_dqn import *

class Timer:
    def __init__(self):
        self._start_time = time.time()

    def __call__(self):
        return time.time() - self._start_time

class Test:
    _fieldnames = [ "N Games", "Dark WR", "Light WR", "Total WR", "Time" ]

    def __init__(self, file_name, timer, ai, enemy_player, n_tests=100):
        self._tester  = [ ai.generate_player, enemy_player ]
        self._n_tests = n_tests
        self._timer   = timer
        self._file    = open('data/{}.csv'.format(file_name), 'w')
        self._writer  = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def __call__(self, game_idx):
        wr = { Disk.dark: 0, Disk.light: 0 }

        for test_idx in range(1, self._n_tests + 1):
            if test_idx % 2:
                dqn_color = Disk.dark
            else:
                dqn_color = Disk.light

            test_game = Reversi(
                self._tester[ (test_idx + 1) % 2 ](delay=0),
                self._tester[  test_idx      % 2 ](delay=0) )
            result = test_game.start_game()

            if result[dqn_color] == Result.win:
                wr[dqn_color] += 1

        test_data = { "N Games": game_idx }
        for color in (Disk.dark, Disk.light):
            test_data[ "{} WR".format(str(color)) ] = wr[color] / (self._n_tests/2)
        test_data["Total WR"] = sum(wr.values()) / self._n_tests
        test_data["Time"] = self._timer()

        self._writer.writerow(test_data)
        for fieldname in self._fieldnames:
            print("{0}: {1}; ".format(fieldname, test_data[fieldname]), end="")
        print()

    def __del__(self):
        self._file.close()

def train(activation_func, n_layers, n_hidden_channels, gpu=0):
    timer = Timer()

    q_func_name = '{activation_func}-{n_layers}x{n_hidden_channels}'.format(
        activation_func   = activation_func,
        n_layers          = n_layers,
        n_hidden_channels = n_hidden_channels )

    ai = [
        ReversiAI(activation_func, n_layers, n_hidden_channels, gpu=gpu),
        ReversiAI(activation_func, n_layers, n_hidden_channels, gpu=gpu) ]
    test = Test(q_func_name, timer, ai[0], Random)

    game_idx = 0
    test(game_idx)

    for game_idx in range(1, 100000 + 1):
        game = Reversi(
            ai[ (game_idx + 1) % 2 ].generate_trainer(),
            ai[  game_idx      % 2 ].generate_trainer() )
        game.start_game()

        if game_idx % 1000 == 0:
            test(game_idx)

        if game_idx in (10000, 50000, 100000):
            for ai_idx, ai_ in enumerate(ai):
                agent_name = '{q_func_name}-{game_idx}-{ai_idx}'.format(
                    q_func_name = q_func_name,
                    game_idx    = game_idx,
                    ai_idx      = ai_idx )
                ai_.agent.save( 'DQN/{}'.format(agent_name) )

    print("total time:", timer())

if __name__ == '__main__':
    train(
        activation_func   = 'leaky_relu',
        n_layers          = 5,
        n_hidden_channels = 256 )
