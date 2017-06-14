import numpy as np


class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros((max_pool_size, observation_dim))
        self._actions = np.zeros((max_pool_size, action_dim))
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

    def add_sample(self, observation, action, reward, terminal, final_state):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self.advance()

    def add_path(self, observations, actions, rewards, terminals, last_obs):
        for t in range(len(observations)):
            self.add_sample(observations[t], actions[t], rewards[t],
                            terminals[t], False)

        self.add_sample(last_obs,
                        np.zeros_like(actions[0]),
                        np.zeros_like(rewards[0]),
                        np.zeros_like(terminals[0]),
                        True)

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


class DoublePool:
    def __init__(self, pool1, pool2, prob1):
        self._pool1 = pool1
        self._pool2 = pool2

        self._p = prob1

    def random_batch(self, batch_size):
        batch1_size = int(batch_size * self._p)
        batch2_size = batch_size - batch1_size

        b1 = self._pool1.random_batch(batch1_size)
        b2 = self._pool1.random_batch(batch2_size)

        return {k: np.concatenate((b1[k], b2[k]), axis=0) for k in b1.keys()}

