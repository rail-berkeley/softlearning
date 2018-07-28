"""Sampler that stores raw actions from policies"""

from .simple_sampler import SimpleSampler


class ExtraPolicyInfoSampler(SimpleSampler):
    def __init__(self, *args, **kwargs):
        super(ExtraPolicyInfoSampler, self).__init__(*args, **kwargs)

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        (action, log_pi, raw_action), _ = self.policy.get_action(
            self._current_observation,
            with_log_pis=True,
            with_raw_actions=True)
        next_observation, reward, terminal, info = self.env.step(action)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            raw_actions=raw_action,
            log_pis=log_pi,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation,
        )

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation
