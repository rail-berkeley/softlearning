"""Implements an ant whose goal is to reach a target in a maze"""

import os

import numpy as np

from rllab.core.serializable import Serializable
from sac.misc.utils import PROJECT_PATH
from .helpers import random_point_in_circle, get_random_goal_logs
from .random_goal_ant_env import RandomGoalAntEnv

MODELS_PATH = os.path.abspath(
    os.path.join(PROJECT_PATH, 'sac/mujoco_models'))

class CrossMazeAntEnv(RandomGoalAntEnv, Serializable):
    """Implements an ant whose goal is to reach a target in a maze"""

    FILE_PATH = os.path.join(MODELS_PATH, 'cross_maze_ant.xml')

    def __init__(self,
                 reward_type='dense',
                 terminate_at_goal=True,
                 goal_reward_weight=3e-1,
                 goal_radius=0.25,
                 goal_distance=5,
                 goal_angle_range=(0, 2*np.pi),
                 velocity_reward_weight=0,
                 ctrl_cost_coeff=1e-2,
                 contact_cost_coeff=1e-3,
                 survive_reward=5e-2,
                 fixed_goal_position=None,
                 *args,
                 **kwargs):
        file_path = self.__class__.FILE_PATH
        kwargs.pop('file_path', None)
        self.fixed_goal_position = fixed_goal_position

        super(CrossMazeAntEnv, self).__init__(
            file_path=file_path,
            reward_type=reward_type,
            terminate_at_goal=terminate_at_goal,
            goal_reward_weight=goal_reward_weight,
            goal_radius=goal_radius,
            goal_distance=goal_distance,
            goal_angle_range=goal_angle_range,
            velocity_reward_weight=velocity_reward_weight,
            ctrl_cost_coeff=ctrl_cost_coeff,
            contact_cost_coeff=contact_cost_coeff,
            survive_reward=survive_reward,
            *args,
            **kwargs)
        self._serializable_initialized = False

    def reset(self, goal_position=None, *args, **kwargs):
        possible_goal_positions = [[6, -6], [6, 6], [12, 0]]

        if goal_position is None:
            if self.fixed_goal_position is not None:
                goal_position = self.fixed_goal_position
            else:
                goal_position = possible_goal_positions[
                    np.random.choice(len(possible_goal_positions))]

        observation = super(CrossMazeAntEnv, self).reset(
            goal_position=np.array(goal_position), *args, **kwargs)

        return observation

    def get_current_obs(self):
        observation = super().get_current_obs()

        if self.fixed_goal_position is not None:
            return observation[:-2]

        return observation

    def render(self, *args, **kwargs):
        result = super(CrossMazeAntEnv, self).render(*args, **kwargs)
        self.viewer.cam.elevation = -55
        self.viewer.cam.lookat[0] = 7
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.9
        self.viewer.cam.azimuth = 0
        self.viewer.cam.trackbodyid = 0

        return result
