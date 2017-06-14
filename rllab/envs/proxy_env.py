from rllab.core.serializable import Serializable


class ProxyEnv(Serializable):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def __getattr__(self, attr):
        if attr == '_wrapped_env':  # Break infinite recursion.
            raise AttributeError

        try:
            return self._wrapped_env.__getattribute__(attr)
        except AttributeError:
            return self._wrapped_env.__getattr__(attr)
