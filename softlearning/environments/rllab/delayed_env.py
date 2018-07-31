import time

from rllab.envs.proxy_env import ProxyEnv
from rllab.core.serializable import Serializable


class DelayedEnv(ProxyEnv, Serializable):
    def __init__(self, env, delay=0.01):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self._delay = delay

    def step(self, action):
        time.sleep(self._delay)
        return self._wrapped_env.step(action)
