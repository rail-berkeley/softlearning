"""Implements a swimmer which is sparsely rewarded for reaching a goal"""

import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger, autoargs

from .helpers import random_point_in_circle, get_random_goal_logs

REWARD_TYPES = ('dense', 'sparse')

class RandomGoalSwimmerEnv(SwimmerEnv):
    """Implements a swimmer which is sparsely rewarded for reaching a goal"""

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(self,
                 reward_type='dense',
                 goal_reward_weight=3e-1,
                 goal_radius=0.25,
                 goal_distance=5,
                 goal_angle_range=(-0.25*np.pi, 0.25*np.pi),
                 velocity_reward_weight=0,
                 terminate_at_goal=True,
                 ctrl_cost_coeff=1e-2,
                 *args,
                 **kwargs):
        assert reward_type in REWARD_TYPES

        self._reward_type = reward_type
        self.terminate_at_goal = terminate_at_goal

        self.goal_reward_weight = goal_reward_weight
        self.goal_radius = goal_radius
        self.goal_distance = goal_distance
        self.goal_angle_range = goal_angle_range

        self.velocity_reward_weight = velocity_reward_weight

        self.ctrl_cost_coeff = ctrl_cost_coeff

        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def reset(self, goal_position=None, *args, **kwargs):
        if goal_position is None:
            goal_position = random_point_in_circle(
                angle_range=self.goal_angle_range,
                radius=self.goal_distance)

        self.goal_position = goal_position

        return super().reset(*args, **kwargs)

    def get_current_obs(self):
        proprioceptive_observation = super().get_current_obs()
        exteroceptive_observation = self.goal_position

        observation = np.concatenate(
            [proprioceptive_observation,
             exteroceptive_observation]
        ).reshape(-1)

        return observation

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)

        done = goal_distance < self.goal_radius

        if self.goal_reward_weight > 0:
            if self._reward_type == 'dense':
                goal_reward = ((np.max(self.goal_distance) - goal_distance)
                               * self.goal_reward_weight)
            elif self._reward_type == 'sparse':
                goal_reward = int(done) * self.goal_reward_weight
        else:
            goal_reward = 0

        if self.velocity_reward_weight > 0:
            xy_velocities = self.get_body_comvel("torso")[:2]
            # rewards for speed on xy-plane (no matter which direction)
            velocity_reward = (self.velocity_reward_weight
                               * np.linalg.norm(xy_velocities))
            if xy_velocities[0] < 0:
                velocity_reward *= -1.0
        else:
            velocity_reward = 0

        # Add control cost
        if self.ctrl_cost_coeff > 0:
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
                np.square(action / scaling))

        else:
            ctrl_cost = 0

        reward = goal_reward + velocity_reward - ctrl_cost
        if not self.terminate_at_goal:
            done = False

        info = {'goal_position': self.goal_position}
        return Step(next_obs, reward, done, **info)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        logs = get_random_goal_logs(paths, self.goal_radius)
        for row in logs:
            logger.record_tabular(*row)
