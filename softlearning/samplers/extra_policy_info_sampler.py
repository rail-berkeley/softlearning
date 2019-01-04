"""Sampler that stores raw actions and log pis from policy."""


from collections import defaultdict

import numpy as np

from .simple_sampler import SimpleSampler


class ExtraPolicyInfoSampler(SimpleSampler):
    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        observations = self.env.convert_to_active_observation(
            self._current_observation)[None]
        actions = self.policy.actions_np([observations])
        log_pis = self.policy.log_pis_np([observations], actions)

        action = actions[0]
        log_pi = log_pis[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self._current_path['observations'].append(self._current_observation)
        self._current_path['actions'].append(action)
        self._current_path['rewards'].append([reward])
        self._current_path['terminals'].append([terminal])
        self._current_path['next_observations'].append(next_observation)
        self._current_path['infos'].append(info)
        # self._current_path['raw_actions'].append(raw_action)
        self._current_path['log_pis'].append(log_pi)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self.policy.reset()
            self._current_observation = self.env.reset()

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return self._current_observation, reward, terminal, info
