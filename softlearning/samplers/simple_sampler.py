import numpy as np

from rllab.misc import logger
from .sampler import Sampler


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        (action, _, _), _ = self.policy.get_action(
            self.env.convert_to_active_observation(self._current_observation))

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            last_path = self.pool.last_n_batch(self._path_length)
            self._last_n_paths.appendleft(last_path)

            self.policy.reset()
            self._current_observation = self.env.reset()

            self._path_length = 0
            self._path_return = 0

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._n_episodes += 1

        else:
            self._current_observation = next_observation

        return self._current_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)
        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)
