"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces, wrappers

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym import register_environments
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from collections import defaultdict


def parse_domain_task(gym_id):
    domain_task_parts = gym_id.split('-')
    domain = '-'.join(domain_task_parts[:1])
    task = '-'.join(domain_task_parts[1:])

    return domain, task


CUSTOM_GYM_ENVIRONMENT_IDS = register_environments()
CUSTOM_GYM_ENVIRONMENTS = defaultdict(list)

for gym_id in CUSTOM_GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    CUSTOM_GYM_ENVIRONMENTS[domain].append(task)

CUSTOM_GYM_ENVIRONMENTS = dict(CUSTOM_GYM_ENVIRONMENTS)

GYM_ENVIRONMENT_IDS = tuple(gym.envs.registry.env_specs.keys())
GYM_ENVIRONMENTS = defaultdict(list)


for gym_id in GYM_ENVIRONMENT_IDS:
    domain, task = parse_domain_task(gym_id)
    GYM_ENVIRONMENTS[domain].append(task)

GYM_ENVIRONMENTS = dict(GYM_ENVIRONMENTS)


class GymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 env=None,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}-{task}"
            env = gym.envs.make(env_id, **kwargs)
        else:
            assert domain is None and task is None, (domain, task)

        if isinstance(env, wrappers.TimeLimit) and unwrap_time_limit:
            # Remove the TimeLimit wrapper that sets 'done = True' when
            # the time limit specified for each environment has been passed and
            # therefore the environment is not Markovian (terminal condition
            # depends on time rather than state).
            env = env.env

        if isinstance(env.observation_space, spaces.Dict):
            observation_keys = (
                observation_keys or list(env.observation_space.spaces.keys()))
        if normalize:
            env = NormalizeActionWrapper(env)

        self._env = env

    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        return observation_space

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        if not isinstance(self._env.observation_space, spaces.Dict):
            return super(GymAdapter, self).active_observation_shape

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        active_size = sum(
            np.prod(self._env.observation_space.spaces[key].shape)
            for key in observation_keys)

        active_observation_shape = (active_size, )

        return active_observation_shape

    def convert_to_active_observation(self, observation):
        if not isinstance(self._env.observation_space, spaces.Dict):
            return observation

        observation_keys = (
            self.observation_keys
            or list(self._env.observation_space.spaces.keys()))

        observation = np.concatenate([
            observation[key] for key in observation_keys
        ], axis=-1)

        return observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = self._env.action_space
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the gym envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
