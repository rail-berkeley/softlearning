"""Implements an adapter for DeepMind Control Suite environments."""

from collections import OrderedDict
import copy

import numpy as np
from dm_control import suite
from dm_env import specs
from dm_control.suite.wrappers import pixels, action_scale
from gym import spaces

from .softlearning_env import SoftlearningEnv


DM_CONTROL_ENVIRONMENTS = {}


def convert_dm_control_to_gym_space(dm_control_space):
    """Recursively convert dm_control_space into gym space.

    Note: Need to check the following cases of the input type, in the following
    order:
       (1) BoundedArraySpec
       (2) ArraySpec
       (3) OrderedDict.

    - Generally, dm_control observation_specs are OrderedDict with other spaces
      (e.g. ArraySpec) nested in it.
    - Generally, dm_control action_specs are of type `BoundedArraySpec`.

    To handle dm_control observation_specs as inputs, we check the following
    input types in order to enable recursive calling on each nested item.
    """
    if isinstance(dm_control_space, specs.BoundedArray):
        shape = dm_control_space.shape
        low = np.broadcast_to(dm_control_space.minimum, shape)
        high = np.broadcast_to(dm_control_space.maximum, shape)
        gym_box = spaces.Box(
            low=low,
            high=high,
            shape=None,
            dtype=dm_control_space.dtype)
        # Note: `gym.Box` doesn't allow both shape and min/max to be defined
        # at the same time. Thus we omit shape in the constructor and verify
        # that it's been implicitly set correctly.
        assert gym_box.shape == dm_control_space.shape, (
            (gym_box.shape, dm_control_space.shape))
        return gym_box
    elif isinstance(dm_control_space, specs.Array):
        if isinstance(dm_control_space, specs.BoundedArray):
            raise ValueError("The order of the if-statements matters.")
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(
                dm_control_space.shape
                if (len(dm_control_space.shape) == 1
                    or (len(dm_control_space.shape) == 3
                        and np.issubdtype(dm_control_space.dtype, np.integer)))
                else (int(np.prod(dm_control_space.shape)), )
            ),
            dtype=dm_control_space.dtype)
    elif isinstance(dm_control_space, OrderedDict):
        return spaces.Dict(OrderedDict([
            (key, convert_dm_control_to_gym_space(value))
            for key, value in dm_control_space.items()
        ]))
    else:
        raise ValueError(dm_control_space)


class DmControlAdapter(SoftlearningEnv):
    """Adapter between SoftlearningEnv and DeepMind Control Suite."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 env=None,
                 rescale_action_range=(-1.0, 1.0),
                 rescale_observation_range=None,
                 observation_keys=(),
                 goal_keys=(),
                 unwrap_time_limit=True,
                 pixel_wrapper_kwargs=None,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")

        if rescale_observation_range:
            raise NotImplementedError(
                "Observation rescaling not implemented for DmControlAdapter.")

        self.rescale_action_range = rescale_action_range
        self.rescale_observation_range = rescale_observation_range

        self.unwrap_time_limit = unwrap_time_limit

        super(DmControlAdapter, self).__init__(
            domain,  task, *args, goal_keys=goal_keys, **kwargs)

        if env is None:
            assert (domain is not None and task is not None), (domain, task)
            env = suite.load(
                domain_name=domain,
                task_name=task,
                task_kwargs=kwargs
                # TODO(hartikainen): Figure out how to pass kwargs to this guy.
                # Need to split into `task_kwargs`, `environment_kwargs`, and
                # `visualize_reward` bool. Check the suite.load(.) in:
                # https://github.com/deepmind/dm_control/blob/master/dm_control/suite/__init__.py
            )
            self._env_kwargs = kwargs
        else:
            assert not kwargs
            assert domain is None and task is None, (domain, task)

        if rescale_action_range:
            should_rescale = (
                np.any(env.action_spec().minimum != rescale_action_range[0])
                or np.any(env.action_spec().maximum != rescale_action_range[1]))
            if should_rescale:
                env = action_scale.Wrapper(
                    env,
                    minimum=(
                        rescale_action_range[0]
                        * np.ones_like(env.action_spec().minimum)),
                    maximum=(
                        rescale_action_range[1]
                        * np.ones_like(env.action_spec().maximum)))
            np.testing.assert_equal(env.action_spec().minimum, -1)
            np.testing.assert_equal(env.action_spec().maximum, 1)

        if pixel_wrapper_kwargs is not None:
            env = pixels.Wrapper(env, **pixel_wrapper_kwargs)

        self._env = env

        assert isinstance(env.observation_spec(), OrderedDict)
        self.observation_keys = (
            observation_keys or tuple(env.observation_spec().keys()))

        observation_space = convert_dm_control_to_gym_space(
            env.observation_spec())

        self._observation_space = type(observation_space)([
            (name, copy.deepcopy(space))
            for name, space in observation_space.spaces.items()
            if name in self.observation_keys + self.goal_keys
        ])

        action_space = convert_dm_control_to_gym_space(self._env.action_spec())

        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(action_space))

        self._action_space = action_space

    def step(self, action, *args, **kwargs):
        time_step = self._env.step(action, *args, **kwargs)
        reward = time_step.reward or 0.0
        terminal = time_step.last()
        info = {
            key: value
            for key, value in time_step.observation.items()
            if key not in self.observation_keys
        }
        observation = self._filter_observation(time_step.observation)
        time_step = time_step._replace(observation=observation)
        return observation, reward, terminal, info

    def reset(self, *args, **kwargs):
        time_step = self._env.reset(*args, **kwargs)
        observation = self._filter_observation(time_step.observation)
        return observation

    def render(self, *args, mode="human", camera_id=0, **kwargs):
        if mode == "human":
            raise NotImplementedError(
                "TODO(Alacarter): Figure out how to not continuously launch"
                " viewers if one is already open."
                " See: https://github.com/deepmind/dm_control/issues/39.")
        elif mode == "rgb_array":
            return self._env.physics.render(
                *args, camera_id=camera_id, **kwargs)

        raise NotImplementedError(mode)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)

    @property
    def unwrapped(self):
        return self._env
