"""Implements an ant whose goal is to reach a target in a maze"""

import os

import numpy as np

from rllab.core.serializable import Serializable
from sac.misc.utils import PROJECT_PATH
from .helpers import random_point_in_circle, get_random_goal_logs
from .random_goal_ant_env import RandomGoalAntEnv

MODELS_PATH = os.path.abspath(
    os.path.join(PROJECT_PATH, 'sac/mujoco_models'))

class SimpleMazeAntEnv(RandomGoalAntEnv, Serializable):
    """Implements an ant whose goal is to reach a target in a maze"""

    FILE_PATH = os.path.join(MODELS_PATH, 'simple_maze_ant.xml')

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
                 *args,
                 **kwargs):
        file_path = self.__class__.FILE_PATH
        kwargs.pop('file_path', None)
        super(SimpleMazeAntEnv, self).__init__(
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

    def reset(self, *args, **kwargs):
        observation = super(SimpleMazeAntEnv, self).reset(
            goal_position=np.array([6, -6]), *args, **kwargs)

        return observation
