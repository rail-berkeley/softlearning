"""Implements an adapter for DeepMind Control Suite environments."""

from collections import OrderedDict

import numpy as np
from dm_control import suite
from dm_control.rl.specs import ArraySpec, BoundedArraySpec
from gym import spaces

from .softlearning_env import SoftlearningEnv


# TODO(hartikainen): Add information of the available environments.
# See `gym_adapter.py` for example.
DM_CONTROL_ENVIRONMENTS = {}


def convert_dm_control_to_gym_space(dm_control_space):
    if isinstance(dm_control_space, BoundedArraySpec):
        gym_box = spaces.Box(
            low=dm_control_space.minimum,
            high=dm_control_space.maximum,
            shape=None,
            dtype=dm_control_space.dtype)
        assert gym_box.shape == dm_control_space.shape, (
            (gym_box.shape, dm_control_space.shape))
        return gym_box
    elif isinstance(dm_control_space, ArraySpec):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype)
    elif isinstance(dm_control_space, OrderedDict):
        return spaces.Dict(OrderedDict([
            (key, convert_dm_control_to_gym_space(value))
            for key, value in dm_control_space.items()
        ]))
    else:
        raise ValueError(dm_control_space)


class DmControlAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for DeepMind Control Suite envs."""

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
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(DmControlAdapter, self).__init__(domain, task, *args, **kwargs)
        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env = suite.load(
                domain_name=domain,
                task_name=task,
                task_kwargs=kwargs
                # TODO(hartikainen): Figure out how to pass kwargs to this guy.
                # Need to split into `task_kwargs`, `environment_kwargs`, and `visualize_reward` bool.
                # Check the suite.load(.) in:
                # https://github.com/deepmind/dm_control/blob/master/dm_control/suite/__init__.py
            )
        else:
            assert domain is None and task is None, (domain, task)

        assert isinstance(env.observation_spec(), OrderedDict)
        self.observation_keys = (
            observation_keys or tuple(env.observation_spec().keys()))
        # TODO(hartikainen): Do we need to normalize actions? For now, assume
        # we don't.

        self._env = env

    @property
    def observation_space(self):
        observation_space = convert_dm_control_to_gym_space(self._env.observation_spec())
        return observation_space

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        observation_space = self.observation_space
        active_size = sum(
            np.prod(observation_space.spaces[key].shape)
            for key in self.observation_keys)
        active_shape = (int(active_size), )
        return active_shape

    def convert_to_active_observation(self, observation):
        flattened_observation = np.concatenate([
            observation[key] for key in self.observation_keys])
        return flattened_observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = convert_dm_control_to_gym_space(self._env.action_spec())
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        timestep = self._env.step(action, *args, **kwargs)
        observation = timestep.observation
        reward = timestep.reward
        terminal = timestep.last()
        info = {}
        #TODO: Find how to extract info
        return observation, reward, terminal, info

    def reset(self, *args, **kwargs):
        timestep = self._env.reset(*args, **kwargs)
        return timestep.observation

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
