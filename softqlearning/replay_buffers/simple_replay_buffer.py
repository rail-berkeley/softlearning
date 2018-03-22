import numpy as np

from rllab.core.serializable import Serializable

from .replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return {
            'observations': self._observations[indices],
            'actions': self._actions[indices],
            'rewards': self._rewards[indices],
            'terminals': self._terminals[indices],
            'next_observations': self._next_obs[indices]
        }

    @property
    def size(self):
        return self._size

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
