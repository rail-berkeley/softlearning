"""Implements a RobosuiteAdapter that converts Robosuite envs into SoftlearningEnv."""

from collections import OrderedDict

import numpy as np
import robosuite as suite
import gym
from gym import spaces

from .softlearning_env import SoftlearningEnv
from softlearning.environments.adapters.gym_adapter import GymAdapter

ROBOSUITE_ENVIRONMENTS = {}

def convert_robosuite_to_gym_obs_space(robosuite_obs_space):
    assert isinstance(robosuite_obs_space, OrderedDict)
    listDict = []
    for key, value in robosuite_obs_space.items():
        listDict.append((key, spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(len(value),),
            dtype=np.float32)))
    return spaces.Dict(OrderedDict(listDict))

def convert_robosuite_to_gym_action_space(robosuite_action_space):
    assert isinstance(robosuite_action_space, tuple)
    return spaces.Box(
        low=robosuite_action_space[0],
        high=robosuite_action_space[1],
        dtype=np.float32)

class RobosuiteAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Robosuite envs."""

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
            "Robosuite environments don't support args. Use kwargs instead.")

        self._Serializable__initialize(locals())

        self.normalize = normalize
        self.unwrap_time_limit = unwrap_time_limit

        super(RobosuiteAdapter, self).__init__(domain, task, *args, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}{task}"
            env = suite.make(env_id, **kwargs)
        else:
            assert domain is None and task is None, (domain, task)

        # TODO(Alacarter): Check how robosuite handles max episode length termination.

        assert isinstance(env.observation_spec(), OrderedDict), env.observation_spec()
        self.observation_keys = observation_keys or tuple(env.observation_spec().keys())
        assert set(self.observation_keys).issubset(set(env.observation_spec().keys())), (
            (self.observation_keys, env.observation_spec().keys()))

        if normalize:
            np.testing.assert_equal(env.action_spec[0], -1, "Ensure spaces are normalized.")
            np.testing.assert_equal(env.action_spec[1], 1, "Ensure spaces are normalized.")

        self._env = env

    @property
    def observation_space(self):
        observation_space = convert_robosuite_to_gym_obs_space(
            self._env.observation_spec())
        return observation_space

    @property
    def active_observation_shape(self):
        """Shape for the active observation based on observation_keys."""
        observation_space = self.observation_space

        active_size = sum(
            np.prod(observation_space.spaces[key].shape)
            for key in self.observation_keys)

        active_observation_shape = (active_size, )

        return active_observation_shape

    def convert_to_active_observation(self, observation):
        observation = np.concatenate([
            observation[key] for key in self.observation_keys
        ], axis=-1)

        return observation

    @property
    def action_space(self, *args, **kwargs):
        action_space = convert_robosuite_to_gym_action_space(self._env.action_spec)
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        return action_space

    def step(self, action, *args, **kwargs):
        # TODO(hartikainen): refactor this to always return an OrderedDict,
        # such that the observations for all the envs is consistent. Right now
        # some of the Robosuite envs return np.array whereas others return dict.
        #
        # Something like:
        # observation = OrderedDict()
        # observation['observation'] = env.step(action, *args, **kwargs)
        # return observation

        return self._env.step(action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        # TODO(Alacarter): Implement rendering so that self._env.viewer.render()
        # can take in args and kwargs
        raise NotImplementedError

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env

    def get_param_values(self, *args, **kwargs):
        raise NotImplementedError

    def set_param_values(self, *args, **kwargs):
        raise NotImplementedError
