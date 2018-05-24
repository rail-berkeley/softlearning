import pickle
import ray  # TODO: Add ray to dependencies.
import tensorflow as tf
import numpy as np

from rllab.misc.overrides import overrides
from rllab.misc import logger

from . import tf_utils
from .sampler import Sampler, rollout


class RemoteSampler(Sampler):
    def __init__(self, **kwargs):
        super(RemoteSampler, self).__init__(**kwargs)

        self._remote_environment = None
        self._remote_path = None
        self._n_episodes = 0
        self._total_samples = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf

    @overrides
    def initialize(self, env, policy, pool):
        super(RemoteSampler, self).initialize(env, policy, pool)

        ray.init()

        env_pkl = pickle.dumps(env)
        policy_pkl = pickle.dumps(policy)

        self._remote_environment = _RemoteEnv.remote(env_pkl, policy_pkl)

    def sample(self):
        if self._remote_path is None:
            policy_params = self.policy.get_param_values()
            self._remote_path = self._remote_environment.rollout.remote(
                policy_params, self._max_path_length)

        path_ready, _ = ray.wait([self._remote_path], timeout=0)

        if len(path_ready) or not self.batch_ready():
            path = ray.get(self._remote_path)
            self.pool.add_path(path)
            self._remote_path = None
            self._total_samples += len(path['observations'])
            self._last_path_return = np.sum(path['rewards'])
            self._max_path_return = max(self._max_path_return,
                                        self._last_path_return)
            self._n_episodes += 1

    def log_diagnostics(self):
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('pool-size', self.pool.size)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


@ray.remote
class _RemoteEnv(object):
    def __init__(self, env_pkl, policy_pkl):
        self._sess = tf_utils.create_session()
        self._sess.run(tf.global_variables_initializer())

        self._env = pickle.loads(env_pkl)
        self._policy = pickle.loads(policy_pkl)

        if hasattr(self._env, 'initialize'):
            self._env.initialize()

    def rollout(self, policy_params, path_length):
        self._policy.set_param_values(policy_params)
        path = rollout(self._env, self._policy, path_length)

        return path
