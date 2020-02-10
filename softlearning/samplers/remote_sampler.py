import pickle
from collections import OrderedDict

import ray
import tensorflow as tf
import numpy as np


from .base_sampler import BaseSampler
from .utils import rollout


class RemoteSampler(BaseSampler):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "TODO(hartikainen): There's a bug here that causes tf to end up in"
            " a RecursionError. This should be fixed/refactored before usage.")
        super(RemoteSampler, self).__init__(**kwargs)

        self._remote_environment = None
        self._remote_path = None
        self._n_episodes = 0
        self._total_samples = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf

    def _create_remote_environment(self, env, policy):
        env_pkl = pickle.dumps(env)
        policy_pkl = pickle.dumps(policy)

        if not ray.is_initialized():
            ray.init()

        self._remote_environment = _RemoteEnv.remote(env_pkl, policy_pkl)

        # Block until the env and policy is ready
        initialized = ray.get(self._remote_environment.initialized.remote())
        assert initialized, initialized

    def initialize(self, environment, policy, pool):
        super(RemoteSampler, self).initialize(environment, policy, pool)
        self._create_remote_environment(environment, policy)

    def wait_for_path(self, timeout=1):
        if self._remote_path is None:
            return [True]

        path_ready, _ = ray.wait([self._remote_path], timeout=timeout)
        return path_ready

    def sample(self, timeout=0):
        if self._remote_path is None:
            policy_params = self.policy.get_weights()
            self._remote_path = self._remote_environment.rollout.remote(
                policy_params, self._max_path_length)

        path_ready = self.wait_for_path(timeout=timeout)

        if len(path_ready) or not self.batch_ready():
            path_samples = ray.get(self._remote_path)
            self._last_n_paths.appendleft(path_samples)

            self.pool.add_samples({
                key: value
                for key, value in path_samples.items()
                if key != 'infos'
            })

            self._remote_path = None
            self._total_samples += path_samples['rewards'].shape[0]
            self._last_path_return = np.sum(path_samples['rewards'])
            self._max_path_return = max(self._max_path_return,
                                        self._last_path_return)
            self._n_episodes += 1

    def get_diagnostics(self):
        diagnostics = OrderedDict({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'pool-size': self.pool.size,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics

    def __getstate__(self):
        super_state = super(RemoteSampler, self).__getstate__()
        state = {
            key: value for key, value in super_state.items()
            if key not in ('_remote_environment', '_remote_path')
        }

        return state

    def __setstate__(self, state):
        super(RemoteSampler, self).__setstate__(state)
        self._remote_path = None


@ray.remote
class _RemoteEnv(object):
    def __init__(self, env_pkl, policy_pkl):
        gpu_options = tf.GPUOptions(allow_growth=True)
        self._session = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))
        tf.compat.v1.keras.backend.set_session(self._session)

        self._env = pickle.loads(env_pkl)
        self._policy = pickle.loads(policy_pkl)

        if hasattr(self._env, 'initialize'):
            self._env.initialize()

        self._initialized = True

    def initialized(self):
        return self._initialized

    def rollout(self, policy_weights, path_length):
        self._policy.set_weights(policy_weights)
        path = rollout(self._env, self._policy, path_length)

        return path
