"""Implements a GymAdapter that converts Gym envs into SoftlearningEnv."""

import numpy as np
import gym
from gym import spaces
from gym.wrappers.dict import FlattenDictWrapper

from .softlearning_env import SoftlearningEnv
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from softlearning.environments.gym.mujoco.ant_env import AntEnv as CustomAntEnv
from softlearning.environments.gym.mujoco.humanoid_env import (
    HumanoidEnv as CustomHumanoidEnv)
from softlearning.environments.gym.mujoco.walker2d_env import (
    Walker2dEnv as CustomWalker2dEnv)
from softlearning.environments.gym.mujoco.hopper_env import (
    HopperEnv as CustomHopperEnv)
from softlearning.environments.gym.mujoco.swimmer_env import (
    SwimmerEnv as CustomSwimmerEnv)
from softlearning.environments.gym.mujoco.pusher_2d_env import (
    Pusher2dEnv,
    ForkReacherEnv)
from softlearning.environments.gym.mujoco.image_pusher import (
    ImagePusherEnv,
    ImageForkReacherEnv,
    BlindForkReacherEnv)
from softlearning.environments.gym.multi_goal import MultiGoalEnv


def raise_on_use_wrapper(e):
    def raise_on_use(*args, **kwargs):
        raise e
    return raise_on_use


try:
    from sac_envs.envs.dclaw.dclaw3_screw_v2 import DClaw3ScrewV2
    from sac_envs.envs.dclaw.dclaw3_screw_v2 import ImageDClaw3Screw
    from sac_envs.envs.dclaw.dclaw3_flip_v1 import DClaw3FlipV1
except ModuleNotFoundError as e:
    DClaw3FlipV1 = raise_on_use_wrapper(e)
    DClaw3ScrewV2 = raise_on_use_wrapper(e)
    ImageDClaw3Screw = raise_on_use_wrapper(e)

try:
    from multiworld.envs.pygame.point2d import Point2DEnv, Point2DWallEnv
except ModuleNotFoundError as e:
    Point2DEnv = raise_on_use_wrapper(e)
    Point2DWallEnv = raise_on_use_wrapper(e)


GYM_ENVIRONMENTS = {
    'swimmer': {
        'default': lambda: gym.envs.make('Swimmer-v2'),
        'custom-default': CustomSwimmerEnv,
    },
    'ant': {
        'default': lambda: gym.envs.make('Ant-v2'),
        'custom-default': CustomAntEnv,
    },
    'humanoid': {
        'default': lambda: gym.envs.make('Humanoid-v2'),
        'standup': lambda: gym.envs.make('HumanoidStandup-v2'),
        'custom-default': CustomHumanoidEnv,
    },
    'hopper': {
        'default': lambda: gym.envs.make('Hopper-v2'),
        'custom-default': CustomHopperEnv,
    },
    'half-cheetah': {
        'default': lambda: gym.envs.make('HalfCheetah-v2')
    },
    'walker': {
        'default': lambda: gym.envs.make('Walker2d-v2'),
        'custom-default': CustomWalker2dEnv,
    },
    'pusher-2d': {
        'default': Pusher2dEnv,
        'default-reach': ForkReacherEnv,

        'image-default': ImagePusherEnv,
        'image-reach': ImageForkReacherEnv,
        'blind-reach': BlindForkReacherEnv,
    },
    'Point2DEnv': {
        'default': Point2DEnv,
        'wall': Point2DWallEnv,
    },
    'HandManipulatePen': {
        'v0': lambda: gym.envs.make('HandManipulatePen-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulatePenDense-v0'),
        'default': lambda: gym.envs.make('HandManipulatePen-v0'),
    },
    'HandManipulateEgg': {
        'v0': lambda: gym.envs.make('HandManipulateEgg-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulateEggDense-v0'),
        'default': lambda: gym.envs.make('HandManipulateEgg-v0'),
    },
    'HandManipulateBlock': {
        'v0': lambda: gym.envs.make('HandManipulateBlock-v0'),
        'Dense-v0': lambda: gym.envs.make('HandManipulateBlockDense-v0'),
        'default': lambda: gym.envs.make('HandManipulateBlock-v0'),
    },
    'HandReach': {
        'v0': lambda: gym.envs.make('HandReach-v0'),
        'Dense-v0': lambda: gym.envs.make('HandReachDense-v0'),
        'default': lambda: gym.envs.make('HandReach-v0'),
    },
    'InvertedDoublePendulum': {
        'v2': lambda: gym.envs.make('InvertedDoublePendulum-v2'),
    },
    'Reacher': {
        'v2': lambda: gym.envs.make('Reacher-v2'),
    },
    'InvertedPendulum': {
        'v2': lambda: gym.envs.make('InvertedPendulum-v2'),
    },
    'DClaw3': {
        'ScrewV2': DClaw3ScrewV2,
        'FlipV1': DClaw3FlipV1,
    },
    'ImageDClaw3': {
        'Screw': ImageDClaw3Screw,
    },
    'HardwareDClaw3': {
        'ScrewV2': lambda *args, **kwargs: (
            DClaw3ScrewV2(
                *args,
                is_hardware=True,
                pose_difference_cost_coeff=kwargs.pop(
                    'pose_difference_cost_coeff', 0),
                joint_velocity_cost_coeff=kwargs.pop(
                    'joint_velocity_cost_coeff', 0),
                joint_acceleration_cost_coeff=kwargs.pop(
                    'joint_acceleration_cost_coeff', 0),
                target_initial_position_range=kwargs.pop(
                    'target_initial_position_range', (np.pi, np.pi)),
                object_initial_position_range=kwargs.pop(
                    'object_initial_position_range', (0, 0)),
                frame_skip=kwargs.pop('frame_skip', 30),
                **kwargs)),
        'ImageScrewV2': lambda *args, **kwargs: (
            ImageDClaw3Screw(
                *args,
                is_hardware=True,
                pose_difference_cost_coeff=kwargs.pop(
                    'pose_difference_cost_coeff', 0),
                joint_velocity_cost_coeff=kwargs.pop(
                    'joint_velocity_cost_coeff', 0),
                joint_acceleration_cost_coeff=kwargs.pop(
                    'joint_acceleration_cost_coeff', 0),
                target_initial_position_range=kwargs.pop(
                    'target_initial_position_range', (np.pi, np.pi)),
                object_initial_position_range=kwargs.pop(
                    'object_initial_position_range', (-np.pi, np.pi)),
                frame_skip=kwargs.pop('frame_skip', 30),
                **kwargs)),
        'FlipV1': lambda *args, **kwargs: (
            DClaw3FlipV1(
                *args,
                is_hardware=True,
                **kwargs)),
    },
    'MultiGoal': {
        'default': MultiGoalEnv
    },
}


class GymAdapter(SoftlearningEnv):
    """Adapter that implements the SoftlearningEnv for Gym envs."""

    def __init__(self,
                 domain,
                 task,
                 *args,
                 normalize=True,
                 observation_keys=None,
                 unwrap_time_limit=True,
                 **kwargs):
        self.normalize = normalize
        self.observation_keys = observation_keys
        self.unwrap_time_limit = unwrap_time_limit

        self._Serializable__initialize(locals())
        super(GymAdapter, self).__init__(domain, task, *args, **kwargs)

        env = GYM_ENVIRONMENTS[domain][task](*args, **kwargs)

        if isinstance(env, gym.wrappers.TimeLimit) and unwrap_time_limit:
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
