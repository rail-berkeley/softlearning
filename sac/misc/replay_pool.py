import numpy as np

from rllab.core.serializable import Serializable


class PoolBase(object):
    def __init__(self, env_spec):
        self._observation_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def action_dim(self):
        return self._action_dim


class SimpleReplayPool(PoolBase, Serializable):
    def __init__(
            self, env_spec, max_pool_size,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        Serializable.quick_init(self, locals())
        super(SimpleReplayPool, self).__init__(env_spec)

        max_pool_size = int(max_pool_size)

        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros((max_pool_size, self._observation_dim))
        self._actions = np.zeros((max_pool_size, self._action_dim))
        self._rewards = np.zeros(max_pool_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        # self._final_state[i] = state i was the final state in a rollout,
        # so it should never be sampled since it has no correspond
        # In other words, we're saving the s_{t+1} after sampling a tuple of
        # (s_t, a_t, r_t, s_{t+1}, TERMINAL=TRUE)
        self._final_state = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0
        self._env_info = dict()

    def add_sample(self, observation, action, reward, terminal, final_state,
                   env_info=None):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state

        if env_info is not None:
            if len(self._env_info) is 0:
                for k, v in env_info.items():
                    self._env_info[k] = np.zeros((self._max_pool_size, v.size))
            for k, v in env_info.items():
                self._env_info[k][self._top] = v

        self.advance()

    def add_path(self, observations, actions, rewards, terminals, last_obs,
                 env_infos=None):

        env_info = None
        if env_infos is not None:
            env_info = dict()
            for k, v in env_infos.items():
                env_info[k] = v[0]

        for t in range(len(observations)):
            if env_info is not None:
                for k, v in env_infos.items():
                    env_info[k] = v[t]

            self.add_sample(observations[t], actions[t], rewards[t],
                            terminals[t], False, env_info)

        self.add_sample(last_obs,
                        np.zeros_like(actions[0]),
                        np.zeros_like(rewards[0]),
                        np.zeros_like(terminals[0]),
                        True,
                        None)

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > 1
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(0, min(self._size, self._max_pool_size))
            # make sure that the transition is valid: if we are at the end of
            # the pool, we need to discard this sample
            if (index + 1) % self._max_pool_size == self._top:
                continue
            # discard the transition if it crosses horizon-triggered resets
            if self._final_state[index]:
                continue
            indices[count] = index
            transition_index = (index + 1) % self._max_pool_size
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(SimpleReplayPool, self).__getstate__()
        if self._env_info is not None:
            env_info_binary = dict()
            for k, v in self._env_info.items():
                env_info_binary[k] = v.tobytes()
            d.update(dict(env_info=env_info_binary))

        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            f=self._final_state.tobytes(),
            bottom=self._bottom,
            top=self._top,
            size=self.size,
        ))

        return d

    def __setstate__(self, d):
        super(SimpleReplayPool, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_pool_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_pool_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_pool_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._final_state = np.fromstring(d['f'], dtype=np.uint8)
        self._bottom = d['bottom']
        self._top = d['top']
        self._size = d['size']

        if 'env_info' in d.keys():
            for k, v in d['env_info'].items():
                self._env_info[k] = np.fromstring(v).reshape(
                    self._max_pool_size, -1
                )
