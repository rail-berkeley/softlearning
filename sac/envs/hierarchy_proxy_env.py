"""Implements an environment proxy to test hierarchy policies"""

from rllab.envs.proxy_env import ProxyEnv
from rllab.core.serializable import Serializable

class HierarchyProxyEnv(ProxyEnv):
    def __init__(self, low_level_policy, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self._low_level_policy = low_level_policy
        super().__init__(*args, **kwargs)

    def step(self, high_level_action):
        current_observation = self._wrapped_env.get_current_obs()

        with self._low_level_policy.fix_h(h=high_level_action[None]):
            action, _ = self._low_level_policy.get_action(
                observation=current_observation[:self._low_level_policy._Ds])

        return super().step(action)
