from abc import abstractmethod

import numpy as np

from rllab.core.serializable import Serializable

from .replay_buffer import ReplayBuffer


class FlexibleReplayPool(ReplayBuffer, Serializable):
    def __init__(self, max_replay_buffer_size, fields):
        super(FlexibleReplayPool, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)
        self._max_buffer_size = max_replay_buffer_size
        self.fields = fields
        self.field_names = list(fields.keys())

        for field_name, field_attrs in fields.items():
            field_shape = [max_replay_buffer_size] + list(field_attrs['shape'])
            setattr(self, field_name, np.zeros(field_shape))

        self._pointer = 0
        self._size = 0

    @property
    def size(self):
        return self._size

    def add_sample(self, **kwargs):
        for field_name in self.field_names:
            getattr(self, field_name)[self._pointer] = kwargs.pop(field_name)

        assert not kwargs, ("Got unexpected fields in the sample: ", kwargs)

        self._advance()

    def __getstate__(self):
        buffer_state = super(FlexibleReplayPool, self).__getstate__()
        buffer_state.update({
            field_name: getattr(self, field_name).tobytes()
            for field_name in self.field_names
        })

        buffer_state.update({
            '_pointer': self._pointer,
            '_size': self._size
        })

        return buffer_state

    def __setstate__(self, buffer_state):
        super(FlexibleReplayPool, self).__setstate__(buffer_state)

        for field_name in self.field_names:
            field = self.fields[field_name]
            flat_values = np.fromstring(
                buffer_state[field_name], dtype=field['dtype'])
            values = flat_values.reshape(
                [self._max_buffer_size] + field['shape'])
            setattr(self, field_name, values)

        self._pointer = buffer_state['_pointer']
        self._size = buffer_state['_size']

    @abstractmethod
    def batch_indices(self, batch_size):
        pass

    def random_batch(self, batch_size, field_name_filter=None):
        field_names = self.field_names
        if field_name_filter is not None:
            field_names = [
                field_name for field_name in field_names
                if field_name_filter(field_name)
            ]

        indices = self.batch_indices(batch_size)

        return {
            field_name: getattr(self, field_name)[indices]
            for field_name in field_names
        }


class SimpleReplayBuffer(FlexibleReplayPool, Serializable):
    def __init__(self, env_spec, *args, **kwargs):
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        fields = {
            'observations': {
                'shape': [self._observation_dim],
                'dtype': 'float32'
            },
            # It's a bit memory inefficient to save the observations twice,
            # but it makes the code *much* easier since you no longer have
            # toworry about termination conditions.
            'next_observations': {
                'shape': [self._observation_dim],
                'dtype': 'float32'
            },
            'actions': {
                'shape': [self._action_dim],
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

        super(SimpleReplayBuffer, self).__init__(*args, fields=fields, **kwargs)

    def terminate_episode(self):
        pass

    def _advance(self):
        self._pointer = (self._pointer + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def batch_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def __getstate__(self):
        buffer_state = super(SimpleReplayBuffer, self).__getstate__()
        buffer_state.update({
            'observations': self._observations.tobytes(),
            'actions': self._actions.tobytes(),
            'rewards': self._rewards.tobytes(),
            'terminals': self._terminals.tobytes(),
            'next_observations': self._next_obs.tobytes(),
            'top': self._top,
            'size': self._size,
        })
        return buffer_state

    def __setstate__(self, buffer_state):
        super(SimpleReplayBuffer, self).__setstate__(buffer_state)

        flat_obs = np.fromstring(buffer_state['observations'])
        flat_next_obs = np.fromstring(buffer_state['next_observations'])
        flat_actions = np.fromstring(buffer_state['actions'])
        flat_reward = np.fromstring(buffer_state['rewards'])
        flat_terminals = np.fromstring(
            buffer_state['terminals'], dtype=np.uint8)

        self._observations = flat_obs.reshape(self._max_buffer_size, -1)
        self._next_obs = flat_next_obs.reshape(self._max_buffer_size, -1)
        self._actions = flat_actions.reshape(self._max_buffer_size, -1)
        self._rewards = flat_reward.reshape(self._max_buffer_size)
        self._terminals = flat_terminals.reshape(self._max_buffer_size)
        self._top = buffer_state['top']
        self._size = buffer_state['size']
