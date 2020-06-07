"""Implements a RobosuiteAdapter that converts Robosuite envs into SoftlearningEnv."""

from collections import OrderedDict
import copy

import numpy as np
import robosuite as suite
from gym import spaces

from .softlearning_env import SoftlearningEnv


ROBOSUITE_ENVIRONMENTS = {}


def convert_robosuite_to_gym_obs_space(robosuite_observation_space):
    assert isinstance(robosuite_observation_space, OrderedDict), type(
        robosuite_observation_space)
    list_dict = []
    for key, value in robosuite_observation_space.items():
        list_dict.append((key, spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=value.shape,
            dtype=value.dtype)))
    return spaces.Dict(OrderedDict(list_dict))


def convert_robosuite_to_gym_action_space(robosuite_action_space):
    assert isinstance(robosuite_action_space, tuple), type(robosuite_action_space)
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
                 rescale_action_range=(-1.0, 1.0),
                 rescale_observation_range=None,
                 observation_keys=(),
                 goal_keys=(),
                 **kwargs):
        assert not args, (
            "Robosuite environments don't support args. Use kwargs instead.")

        if rescale_observation_range:
            raise NotImplementedError(
                "Observation rescaling not implemented for RobosuiteAdapter.")

        self.rescale_action_range = rescale_action_range
        self.rescale_observation_range = rescale_observation_range

        super(RobosuiteAdapter, self).__init__(
            domain, task, *args, goal_keys=goal_keys, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env_id = f"{domain}{task}"
            env = suite.make(env_id, **kwargs)
            self._env_kwargs = kwargs
        else:
            assert not kwargs, kwargs
            assert domain is None and task is None, (domain, task)

        # TODO(Alacarter): Check how robosuite handles max episode length
        # termination.

        observation_spec = env.observation_spec()
        assert isinstance(observation_spec, OrderedDict), observation_spec
        self.observation_keys = (
            observation_keys or tuple(observation_spec.keys()))
        assert set(self.observation_keys).issubset(
            set(observation_spec.keys())
        ), (self.observation_keys, observation_spec.keys())

        if rescale_action_range:
            np.testing.assert_equal(
                env.action_spec,
                rescale_action_range,
                "Ensure spaces are normalized.")

        self._env = env

        observation_space = convert_robosuite_to_gym_obs_space(
            self._env.observation_spec())

        self._observation_space = type(observation_space)([
            (name, copy.deepcopy(space))
            for name, space in observation_space.spaces.items()
            if name in self.observation_keys + self.goal_keys
        ])

        action_space = convert_robosuite_to_gym_action_space(
            self._env.action_spec)

        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(action_space))

        self._action_space = action_space

    def step(self, action, *args, **kwargs):
        observation, reward, terminal, info = self._env.step(
            action, *args, **kwargs)

        observation = self._filter_observation(observation)

        return observation, reward, terminal, info

    def reset(self, *args, **kwargs):
        observation = self._env.reset(*args, **kwargs)

        observation = self._filter_observation(observation)

        return observation

    def render(self,
               *args,
               mode="human",
               camera_id=None,
               camera_name=None,
               width=None,
               height=None,
               depth=None,
               **kwargs):
        if mode == "human":
            return self._env.render(*args, **kwargs)
        elif mode == "rgb_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            if camera_id is not None:
                camera_name = self.sim.model.camera_id2name(camera_id)

            pixels = self._env.sim.render(
                camera_name=camera_name or self._env.camera_name,
                width=width or self._env.camera_width,
                height=height or self._env.camera_height,
                depth=depth or self._env.camera_depth
            )[::-1]
            return pixels

        raise NotImplementedError(mode)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    def copy(self):
        """Override default copy method to allow robosuite env serialization.

        Robosuite environments are not serializable, and thus we cannot use the
        default `copy.deepcopy(self)` from `SoftlearningEnv`. Instead, we first
        create a copy of the self *without* robosuite environment (`self._env`)
        and then instantiate a new robosuite environment and attach it to the
        copied self.
        """
        env = self._env
        self._env = None
        result = copy.deepcopy(self)
        result._env = suite.make(
            f"{self._domain}{self._task}", **self._env_kwargs)
        self._env = env

        return result

    @property
    def unwrapped(self):
        return self._env

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key != '_env'
        }
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._env = suite.make(
            f"{self._domain}{self._task}", **self._env_kwargs)
