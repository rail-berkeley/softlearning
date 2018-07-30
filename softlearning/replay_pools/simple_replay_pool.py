import numpy as np

from rllab.core.serializable import Serializable

from .replay_pool import ReplayPool
from .flexible_replay_pool import FlexibleReplayPool


class SimpleReplayPool(FlexibleReplayPool, Serializable):
    def __init__(self, observation_shape, action_shape, *args, **kwargs):
        Serializable.quick_init(self, locals())

        self._observation_shape = observation_shape
        self._action_shape = action_shape

        fields = {
            'observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            # It's a bit memory inefficient to save the observations twice,
            # but it makes the code *much* easier since you no longer have
            # toworry about termination conditions.
            'next_observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            'actions': {
                'shape': self._action_shape,
                'dtype': 'float32'
            },
            'rewards': {
                'shape': [],
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': [],
                'dtype': 'bool'
            },
        }

        super(SimpleReplayPool, self).__init__(*args, fields=fields, **kwargs)

    def terminate_episode(self):
        pass

    def _advance(self):
        self._pointer = (self._pointer + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def batch_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def __getstate__(self):
        pool_state = super(SimpleReplayPool, self).__getstate__()
        pool_state.update({
            'observations': self.observations.tobytes(),
            'actions': self.actions.tobytes(),
            'rewards': self.rewards.tobytes(),
            'terminals': self.terminals.tobytes(),
            'next_observations': self.next_observations.tobytes(),
            'pointer': self._pointer,
            'size': self._size,
        })
        return pool_state

    def __setstate__(self, pool_state):
        super(SimpleReplayPool, self).__setstate__(pool_state)

        flat_obs = np.fromstring(pool_state['observations'])
        flat_next_obs = np.fromstring(pool_state['next_observations'])
        flat_actions = np.fromstring(pool_state['actions'])
        flat_reward = np.fromstring(pool_state['rewards'])
        flat_terminals = np.fromstring(
            pool_state['terminals'], dtype=np.uint8)

        self.observations = flat_obs.reshape(self._max_size, -1)
        self.next_observations = flat_next_obs.reshape(self._max_size, -1)
        self.actions = flat_actions.reshape(self._max_size, -1)
        self.rewards = flat_reward.reshape(self._max_size)
        self.terminals = flat_terminals.reshape(self._max_size)
        self._pointer = pool_state['pointer']
        self._size = pool_state['size']
