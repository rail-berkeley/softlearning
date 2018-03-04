"""Implements an ant whose goal is to reach target in a maze"""

import os

import numpy as np

from rllab.core.serializable import Serializable
from sac.misc.utils import PROJECT_PATH
from .helpers import random_point_in_circle, get_random_goal_logs
from .random_goal_ant_env import RandomGoalAntEnv

MODELS_PATH = os.path.abspath(os.path.join(PROJECT_PATH, 'models'))

class SimpleMazeAntEnv(RandomGoalAntEnv, Serializable):
    """Implements an ant whose goal is to reach target in a maze"""

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
        file_path = kwargs.pop('file_path', self.FILE_PATH)
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

    def get_current_obs(self):
        proprioceptive_observation = np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext[:14, ...], -1, 1).flat,
            self.get_body_xmat('torso').flat,
            self.get_body_com('torso'),
        ]).reshape(-1)

        if self.goal_reward_weight > 0:
            exteroceptive_observation = self.goal_position
        else:
            exteroceptive_observation = np.zeros_like(self.goal_position)

        observation = np.concatenate(
            [proprioceptive_observation,
             exteroceptive_observation]
        ).reshape(-1)

        return observation

    def reset(self, goal_position=None, *args, **kwargs):
        observation = super().reset(goal_position, *args, **kwargs)

        wall_names = ['wall-{}'.format(i) for i in range(4)]
        wall_idx = [self.model.geom_names.index(wall_name) for wall_name in wall_names]
        hide_idx = np.random.choice(wall_idx, 2, replace=False)
        wall_positions = [
            (0, 5, 0.5),
            (5, 0, 0.5),
            (0, -5, 0.5),
            (-5, 0, 0.5),
        ]

        new_geom_pos = self.model.geom_pos.copy()
        new_geom_pos[wall_idx, ...] = wall_positions
        new_geom_pos[hide_idx, 2] = 20 # raise the wall in the air
        self.model.geom_pos = new_geom_pos

        return observation
