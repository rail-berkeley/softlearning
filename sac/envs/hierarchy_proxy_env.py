"""Implements an environment proxy to test hierarchy policies"""

from rllab.envs.normalized_env import NormalizedEnv

class HierarchyProxyEnv(object):
    def __init__(self, env, low_level_policy, *args, **kwargs):
        self._env = env
        self._low_level_policy = low_level_policy

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, high_level_action):
        current_observation = (
            self._env.wrapped_env.get_current_obs()
            if isinstance(self._env, NormalizedEnv)
            else self._env.get_current_obs())

        with self._low_level_policy.fix_h(h=high_level_action[None]):
            action = self._low_level_policy.get_action(
                observation=current_observation[:self._low_level_policy._Ds])

        return self._env.step(action)
