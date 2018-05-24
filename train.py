import sys
import os
import shutil
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

class Test:
    def __init__(self, tester_gen, enemy_gen, n_tests):
        self._tester_gen = tester_gen
        self._enemy_gen  = enemy_gen
        self._n_tests    = n_tests

    def __call__(self):
        n_wins = { Disk.dark: 0, Disk.light: 0 }

        for test_idx in range(1, self._n_tests + 1):
            players = ( self._tester_gen(), self._enemy_gen() )

            game = Reversi(
                players[(test_idx + 1) % 2],
                players[ test_idx      % 2] )
            result = game.start_game()

            if result[players[0].color] == Result.win:
                n_wins[players[0].color] += 1

        data = {}
        data["Dark_WR" ] = n_wins[Disk.dark ] / (self._n_tests / 2)
        data["Light_WR"] = n_wins[Disk.light] / (self._n_tests / 2)
        data["Total_WR"] = sum(n_wins.values()) / self._n_tests
        return data

class Train:
    status_log_fields = ("n_games", "dark_wr", "light_wr", "total_wr", "average_q", "average_loss")
    test_log_fields   = ("N_Games", "Dark_WR", "Light_WR", "Total_WR", "Time")

    def __init__(self, ai, enemy_gen, n_games, tester_gen, n_tests, test_log_interval,
            status_log_interval, ai_name, need_save=lambda game_idx, log: False, append_output=False):
        if isinstance(ai, (tuple, list)):
            self._ai = ai[0]
            self._ai_list = ai
        else:
            self._ai = ai
            self._ai_list = None

        self._trainer_gen = self._ai.generate_trainer
        self._enemy_gen = enemy_gen
        self._n_games = n_games
        self._test = Test(self._ai.generate_player, tester_gen, n_tests)
        self._test_log_interval = test_log_interval
        self._status_log_interval = status_log_interval
        self._save_dir = "DQN/" + ai_name + "/"
        self._log_dir = "log/" + ai_name + "/"
        self._need_save = need_save

        if not append_output:
            Train.confirm_overwrite(self._log_dir)
            Train.confirm_overwrite(self._save_dir)
        os.makedirs(self._log_dir, exist_ok=True)

        self._timer  = Timer()
        self._outctl = OutputController()

    @staticmethod
    def confirm_overwrite(dir):
        if os.path.isdir(dir) and os.listdir(dir):
            overwrite = input(dir + " already exists. Overwrite it? (y/n): ") == 'y'
            if overwrite:
                shutil.rmtree(dir)
            else:
                sys.exit()

    def __call__(self):
        self._timer.start()
        test_log = self.log_test(self._test, 0, "test")
        self._outctl.output_test_result(test_log)
        res_log = { color: { Result.win: 0, Result.lose: 0, Result.draw: 0 }
            for color in (Disk.dark, Disk.light) }

        for game_idx in range(1, self._n_games + 1):
            self._outctl.update_game_idx(game_idx)
            players = ( self._trainer_gen(), self._enemy_gen() )

            game = Reversi(
                players[(game_idx + 1) % 2],
                players[ game_idx      % 2] )
            result = game.start_game()

            color = players[0].color
            res = result[color]
            if res == Result.win:
                res_log[color][Result.win ] += 1
            elif res == Result.lose:
                res_log[color][Result.lose] += 1
            else:
                res_log[color][Result.draw] += 1

            if game_idx % self._status_log_interval == 0:
                self.log_status_all_ai(game_idx, res_log)
                res_log = { color: { Result.win: 0, Result.lose: 0, Result.draw: 0 }
                    for color in (Disk.dark, Disk.light) }

            if game_idx % self._test_log_interval == 0:
                test_log = self.log_test(self._test, game_idx, "test")
                self._outctl.output_test_result(test_log)

            if self._need_save(game_idx, test_log):
                if self._ai_list is None:
                    self._ai.agent.save(self._save_dir + str(game_idx))
                else:
                    for ai_idx, ai in enumerate(self._ai_list):
                        ai.agent.save(self._save_dir + str(game_idx) + "-" + str(ai_idx))

    def log_status_all_ai(self, game_idx, result):
        if self._ai_list is None:
            res = { color: result[color][Result.win] / sum(result[color].values())
                for color in (Disk.dark, Disk.light) }
            self.log_status(self._ai, game_idx, res, "status")
        else:
            res = (
                { color: result[color][Result.win] / sum(result[color].values())
                    for color in (Disk.dark, Disk.light) },
                { color: result[color.reverse()][Result.lose] / sum(result[color.reverse()].values())
                    for color in (Disk.dark, Disk.light) } )
            for i in range(2):
                self.log_status(self._ai_list[i], game_idx, res[i], "status-" + str(i))

    def log_status(self, ai, game_idx, win_rate, file_name):
        file_path = self._log_dir + file_name + ".csv"

        if not os.path.isfile(file_path):
            with open(file_path, 'x') as file:
                writer = csv.DictWriter(file, Train.status_log_fields)
                writer.writeheader()

        with open(file_path, 'a') as file:
            writer = csv.DictWriter(file, Train.status_log_fields)
            log = dict(ai.agent.get_statistics())
            log["n_games"]  = game_idx
            log["dark_wr"]  = win_rate[Disk.dark]
            log["light_wr"] = win_rate[Disk.light]
            log["total_wr"] = sum(win_rate.values()) / 2
            writer.writerow(log)

        return log

    def log_test(self, test, game_idx, file_name):
        file_path = self._log_dir + file_name + ".csv"

        if not os.path.isfile(file_path):
            with open(file_path, 'x') as file:
                writer = csv.DictWriter(file, Train.test_log_fields)
                writer.writeheader()

        with open(file_path, 'a') as file:
            writer = csv.DictWriter(file, Train.test_log_fields)
            log = test()
            log["N_Games"] = game_idx
            log["Time"] = self._timer.get_time()
            writer.writerow(log)

        return log

class OutputController:
    def __init__(self):
        self._idx_updating = False

        sys.stderr.write("\n")
        sys.stderr.flush()
        print("    ".join(("{:<9}".format(label) for label in Train.test_log_fields)))

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
            n_games = result["N_Games"],
            dark    = result["Dark_WR"],
            light   = result["Light_WR"],
            total   = result["Total_WR"],
            time    = "{:>3}:{:0>2}:{:0>2}".format(hour, minute, second) ))

def generate_ai(activation_func, n_layers, n_hidden_channels,
        gamma         = 0.95,
        start_epsilon = 1.0,
        end_epsilon   = 0.3,
        decay_steps   = 50000,
        enemy         = 'RND',
        gpu           = 0 ):

    ai = ReversiAI(activation_func, n_layers, n_hidden_channels, gamma=gamma,
        start_epsilon=start_epsilon, end_epsilon=end_epsilon, decay_steps=decay_steps, gpu=gpu)

    if enemy == 'RND':
        enemy_gen = Random
    elif enemy == 'DQN':
        enemy_ai = ReversiAI(activation_func, n_layers, n_hidden_channels, decay_steps=decay_steps, gpu=gpu)
        enemy_gen = enemy_ai.generate_trainer
        ai = ( ai, enemy_ai )
    elif enemy == 'SLFP':
        enemy_gen = ai.generate_player
    elif enemy == 'SLFT':
        enemy_gen = ai.generate_trainer

    ai_name = '{activation_func}-{n_layers}x{n_hidden_channels}'.format(
        activation_func   = activation_func,
        n_layers          = n_layers,
        n_hidden_channels = n_hidden_channels )

    ai_name += '-vs' + enemy

    return ai, ai_name, enemy_gen

def main():
    ai, ai_name, enemy_gen = generate_ai(
        activation_func   = 'leaky_relu',
        n_layers          = 5,
        n_hidden_channels = 256,
        gamma             = 0.95,
        start_epsilon     = 1.0,
        end_epsilon       = 0.3,
        decay_steps       = 50000,
        enemy             = 'SLFT' )

    train = Train(
        ai                  = ai,
        enemy_gen           = enemy_gen,
        n_games             = 50000,
        tester_gen          = Random,
        n_tests             = 200,
        test_log_interval   = 1000,
        status_log_interval = 100,
        ai_name             = ai_name,
        need_save = lambda i, log: i == log["N_Games"] and log["Total_WR"] >= 0.9 )

    train()

if __name__ == '__main__':
    main()
