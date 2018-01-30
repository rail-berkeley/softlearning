import numpy as np
import time
import ray
import pickle

import tensorflow as tf

from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc import tf_utils


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        return self.pool.size >= self._min_pool_size

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        raise NotImplementedError


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

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

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

    def log_diagnostics(self):
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('pool-size', self.pool.size)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


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
