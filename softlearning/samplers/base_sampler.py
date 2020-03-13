from collections import deque, OrderedDict
from itertools import islice


class BaseSampler(object):
    def __init__(self,
                 max_path_length,
                 environment=None,
                 policy=None,
                 pool=None,
                 store_last_n_paths=10):
        self._max_path_length = max_path_length
        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self.environment = environment
        self.policy = policy
        self.pool = pool

    def initialize(self, environment, policy, pool):
        self.environment = environment
        self.policy = policy
        self.pool = pool

    def reset(self):
        pass

    def set_policy(self, policy):
        self.policy = policy

    def clear_last_n_paths(self):
        self._last_n_paths.clear()

    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    def sample(self):
        raise NotImplementedError

    def terminate(self):
        self.environment.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({'pool-size': self.pool.size})
        return diagnostics

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key not in (
                    'environment',
                    'policy',
                    'pool',
                    '_last_n_paths',
                    '_current_observation',
                    '_current_path',
                    '_is_first_step',
            )
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.environment = None
        self.policy = None
        self.pool = None
        # TODO(hartikainen): Maybe try restoring these from the pool?
        self._last_n_paths = deque(maxlen=self._store_last_n_paths)
