import time
import numpy as np
from cached_property import cached_property
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import gym.spaces
from gym.spaces import prng
from reversi_engine import *

class ReversiEnv(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(64)
        self.observation_space = gym.spaces.Box(
            low = np.zeros(8*8*4, dtype=np.float32),
            high = np.ones(8*8*4, dtype=np.float32) )

    def reset(self, player):
        self._player = player

    @TailRecursive()
    def call_step(self, action):
        row = action // 8
        col = action % 8
        return self._player.game.tell_where_put(row, col)

    def return_step(self, done, result=None):
        if done:
            if result == Result.win:
                reward = 1
            elif result == Result.lose:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self._get_observation(), reward

    def _get_observation(self):
        self._possible_place = np.array(self._player.game.board.possible_place, dtype=np.bool).ravel()
        board = np.array( [
            elm
            for array_row in self._player.game.board.array[0:8]
                for disk in array_row[0:8]
                    for elm in self._disk_to_vector(disk) ], dtype=np.float32 )

        return np.hstack( (self._possible_place.astype(np.float32), board) )

    def _disk_to_vector(self, disk):
        if disk == self._player.color:
            return [1, 0, 0]
        elif disk == self._player.color.reverse():
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def action_space_sample(self):
        place_list = np.where(self._possible_place)[0].astype(np.int32)
        return prng.np_random.choice(place_list)

class ReversiActionValue(chainerrl.action_value.DiscreteActionValue):
    @classmethod
    def setup_dummy_action(cls, xp):
        cls._dummy_action_col = xp.zeros(64, dtype=np.bool)
        cls._dummy_action_col[27] = True

    def __init__(self, q_values, possible_places, q_values_formatter=lambda x: x):
        super().__init__(q_values, q_values_formatter)
        self._impossible_places = ~possible_places.astype(np.bool)

    @cached_property
    def greedy_actions(self):
        q_val = self.q_values.data.copy()
        q_val[ self._impossible_places ] = -np.inf

        # currently, CuPy only supports slices that consist of one boolean array.
        #q_val[ self._impossible_places.all(axis=1), 27 ] = 0.0
        q_val[ self._impossible_places.all(axis=1).reshape((-1,1)) * self._dummy_action_col ] = 0.0

        return chainer.Variable( q_val.argmax(axis=1).astype(np.int32) )

class QFunction(chainer.Chain):
    def __init__(self, activation_func, n_layers, obs_size, n_hidden_channels, n_actions):
        self._activation_func = activation_func
        self._n_actions       = n_actions

        self._layers = [ L.Linear(obs_size, n_hidden_channels) ]
        for idx in range(1, n_layers-1):
            self._layers.append( L.Linear(n_hidden_channels, n_hidden_channels) )
        self._layers.append( L.Linear(n_hidden_channels, n_actions) )

        super().__init__( **{ "l{}".format(idx): layer for idx, layer in enumerate(self._layers) } )

    def __call__(self, x, test=False):
        h = self.l0(x)
        for layer in self._layers[1:]:
            h = layer( self._activation_func(h) )
        return ReversiActionValue( h, x[:,0:self._n_actions] )

class ReversiDQN(chainerrl.agents.DoubleDQN):
    def __init__(self, env, activation_func, n_layers, n_hidden_channels,
        gpu=None, gamma=0.95, start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000):

        obs_size  = env.observation_space.shape[0]
        n_actions = env.action_space.n
        q_func    = QFunction(activation_func, n_layers, obs_size, n_hidden_channels, n_actions)

        optimizer = chainer.optimizers.Adam()
        optimizer.setup(q_func)

        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)

        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            start_epsilon      = start_epsilon,
            end_epsilon        = end_epsilon,
            decay_steps        = decay_steps,
            random_action_func = env.action_space_sample )

        super().__init__(q_func, optimizer, replay_buffer, gamma, explorer, gpu=gpu)

        ReversiActionValue.setup_dummy_action(self.xp)

class ReversiAI:
    def __init__(self, activation_func, n_layers, n_hidden_channels,
        gpu=None, gamma=0.95, start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000):

        self.env = ReversiEnv()
        activation_func_ = eval('F.{}'.format(activation_func))
        self.agent = ReversiDQN( self.env, activation_func_, n_layers, n_hidden_channels,
            gpu           = gpu,
            gamma         = gamma,
            start_epsilon = start_epsilon,
            end_epsilon   = end_epsilon,
            decay_steps   = decay_steps )

    def generate_trainer(self):
        return DQNTrainer(self)

    def generate_player(self, delay=0.5):
        return DQNPlayer(self, delay)

class DQNTrainer(Player):
    def __init__(self, ai):
        self._ai = ai
        self._ai.env.reset(self)

    @TailRecursive()
    def tell_your_turn(self):
        obs, reward = self._ai.env.return_step(False)
        action = self._ai.agent.act_and_train(obs, reward)
        return self._ai.env.call_step(action)

    def tell_game_result(self, result):
        obs, reward = self._ai.env.return_step(True, result)
        self._ai.agent.stop_episode_and_train(obs, reward, True)

class DQNPlayer(Player):
    def __init__(self, ai, delay=0.5):
        self._ai = ai
        self._ai.env.reset(self)
        self._delay = delay

    @TailRecursive()
    def tell_your_turn(self):
        #time.sleep(self._delay)

        obs, reward = self._ai.env.return_step(False)
        action = self._ai.agent.act(obs)
        return self._ai.env.call_step(action)

    def tell_game_result(self, result):
        obs, reward = self._ai.env.return_step(True, result)
        self._ai.agent.stop_episode()

        print( "{0} DQN {1}!".format(str(self.color), str(result)) )