from functools import wraps
from cached_property import cached_property
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import gym.spaces
from gym.spaces import prng
from engine import *

#from debug import *

class ReversiEnv(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(64)
        self.observation_space = gym.spaces.Box(
            low = np.zeros(8*8*4, dtype=np.float32),
            high = np.ones(8*8*4, dtype=np.float32),
            dtype = np.float32 )

    def reset(self, game):
        self.call_step = game.tail_recursive(self._call_step)
        self._game = game

    def _call_step(self, action):
        row, col = divmod(action, 8)
        return self._game.tell_where_put(row, col)

    def return_step(self, color, done, result=None):
        if done:
            if result is Result.win:
                reward = 1
            elif result is Result.lose:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self._get_observation(color), reward

    def _get_observation(self, color):
        self._possible_place = np.array(self._game.board.possible_place, dtype=np.bool).ravel()
        board = np.array( [
            elm
            for array_row in self._game.board.array[0:8]
                for disk in array_row[0:8]
                    for elm in ReversiEnv._disk_to_vector(color, disk) ], dtype=np.float32 )

        return np.hstack( (self._possible_place.astype(np.float32), board) )

    @staticmethod
    def _disk_to_vector(color, disk):
        if disk is color:
            return (1, 0, 0)
        elif disk is color.reverse():
            return (0, 1, 0)
        else:
            return (0, 0, 1)

    def action_space_sample(self):
        place_list = np.where(self._possible_place)[0].astype(np.int32)
        return prng.np_random.choice(place_list)

class ReversiActionValue(chainerrl.action_value.DiscreteActionValue):
    def __init__(self, q_values, possible_places, q_values_formatter=lambda x: x):
        super().__init__(q_values, q_values_formatter)
        self._impossible_places = ~possible_places.astype(np.bool)

    @cached_property
    def greedy_actions(self):
        q_val = self.q_values.data.copy()
        q_val[ self._impossible_places ] = -np.inf
        return chainer.Variable( q_val.argmax(axis=1).astype(np.int32) )

class QFunction(chainer.Chain):
    def __init__(self, activation_func, n_layers, obs_size, n_hidden_channels, n_actions):
        self._activation_func = activation_func
        self._n_actions       = n_actions

        self._layers = [ L.Linear(obs_size, n_hidden_channels) ]
        for idx in range(1, n_layers-1):
            self._layers.append( L.Linear(n_hidden_channels, n_hidden_channels) )
        self._layers.append( L.Linear(n_hidden_channels, n_actions) )

        super().__init__( **{ 'l{}'.format(idx): layer for idx, layer in enumerate(self._layers) } )

    #@output_q_vals
    def __call__(self, x, test=False):
        h = self.l0(x)
        for layer in self._layers[1:]:
            h = layer( self._activation_func(h) )
        return ReversiActionValue( h, x[:,0:self._n_actions] )

class ReversiDQN(chainerrl.agents.DoubleDQN):
    def __init__(self, env, activation_func, n_layers, n_hidden_channels,
        gpu, gamma, start_epsilon, end_epsilon, decay_steps):

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

        self.last_state_of  = { Disk.dark:None, Disk.light:None }
        self.last_action_of = { Disk.dark:None, Disk.light:None }

    @staticmethod
    def _distinguish_player(func):
        @wraps(func)
        def _func(self, color, *args, **kwargs):
            self.last_state  = self.last_state_of[color]
            self.last_action = self.last_action_of[color]
            result = func(self, *args, **kwargs)
            self.last_state_of[color]  = self.last_state
            self.last_action_of[color] = self.last_action
            return result
        return _func

    @_distinguish_player.__func__
    def act_and_train(self, obs, reward):
        return super().act_and_train(obs, reward)

    @_distinguish_player.__func__
    def stop_episode_and_train(self, state, reward, done=None):
        super().stop_episode_and_train(state, reward, done)

    @_distinguish_player.__func__
    def stop_episode_(self):
        self.stop_episode()

class ReversiAI:
    def __init__( self, activation_func, n_layers, n_hidden_channels,
        gamma         = 0.95,
        start_epsilon = 1.0,
        end_epsilon   = 0.3,
        decay_steps   = 50000,
        gpu           = None ):

        self.env = ReversiEnv()
        activation_func_ = eval('F.{}'.format(activation_func))
        self.agent = ReversiDQN( self.env, activation_func_, n_layers, n_hidden_channels,
            gpu, gamma, start_epsilon, end_epsilon, decay_steps )

    def generate_trainer(self):
        return DQNTrainer(self)

    def generate_player(self):
        return DQNPlayer(self)

class DQNTrainer(Player):
    def __init__(self, ai):
        self._ai = ai

    def setup_player(self, game, color):
        self._ai.env.reset(game)
        return super().setup_player(game, color)

    def tell_your_turn(self):
        obs, reward = self._ai.env.return_step(self.color, False)
        action = self._ai.agent.act_and_train(self.color, obs, reward)
        return self._ai.env.call_step(action)

    def tell_game_result(self, result):
        obs, reward = self._ai.env.return_step(self.color, True, result)
        self._ai.agent.stop_episode_and_train(self.color, obs, reward, True)

class DQNPlayer(Player):
    def __init__(self, ai):
        self._ai = ai

    def setup_player(self, game, color):
        self._ai.env.reset(game)
        return super().setup_player(game, color)

    def tell_your_turn(self):
        obs, reward = self._ai.env.return_step(self.color, False)
        action = self._ai.agent.act(obs)
        return self._ai.env.call_step(action)

    def tell_game_result(self, result):
        obs, reward = self._ai.env.return_step(self.color, True, result)
        self._ai.agent.stop_episode_(self.color)

        #print( "{} DQN {}!".format(str(self.color), str(result)) )
