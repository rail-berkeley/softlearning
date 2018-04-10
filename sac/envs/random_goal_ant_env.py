"""Implements a ant which is sparsely rewarded for reaching a goal"""

import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger, autoargs

from .helpers import random_point_in_circle, get_random_goal_logs


REWARD_TYPES = ('dense', 'sparse')

class RandomGoalAntEnv(AntEnv):
    """Implements a ant env which is sparsely rewarded for reaching a goal"""

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    @autoargs.arg('survive_reward', type=float,
                  help='bonus reward for being alive')
    @autoargs.arg('contact_cost_coeff', type=float,
                  help='cost coefficient for contact')
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
        assert reward_type in REWARD_TYPES

        self._reward_type = reward_type
        self.terminate_at_goal = terminate_at_goal

        self.goal_reward_weight = goal_reward_weight
        self.goal_radius = goal_radius
        self.goal_distance = goal_distance
        self.goal_angle_range = goal_angle_range

        self.velocity_reward_weight = velocity_reward_weight

        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.contact_cost_coeff = contact_cost_coeff
        self.survive_reward = survive_reward

        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def reset(self, goal_position=None, *args, **kwargs):
        if goal_position is None:
            goal_position = random_point_in_circle(
                angle_range=self.goal_angle_range,
                radius=self.goal_distance)

        self.goal_position = goal_position

        if 'target' in self.model.geom_names:
            goal_geom_idx = self.model.geom_names.index('target')

            new_geom_pos = self.model.geom_pos.copy()
            new_geom_pos[goal_geom_idx] = np.concatenate([goal_position, [0]])
            self.model.geom_pos = new_geom_pos

            new_geom_size = self.model.geom_size.copy()
            new_geom_size[goal_geom_idx] = np.array([self.goal_radius, 0, 0])
            self.model.geom_size = new_geom_size

        return super().reset(*args, **kwargs)

    def get_current_obs(self):
        proprioceptive_observation = super().get_current_obs()
        if self.goal_reward_weight > 0:
            exteroceptive_observation = self.goal_position
        else:
            exteroceptive_observation = np.zeros_like(self.goal_position)

        observation = np.concatenate(
            [proprioceptive_observation,
             exteroceptive_observation]
        ).reshape(-1)

        return observation

    @overrides
    def step(self, action):
        self.forward_dynamics(action)

        xy_position = self.get_body_com('torso')[:2]
        goal_distance = np.linalg.norm(xy_position - self.goal_position)

        goal_reached = goal_distance < self.goal_radius

        if self.goal_reward_weight > 0:
            if self._reward_type == 'dense':
                goal_reward = ((np.max(self.goal_distance) - goal_distance)
                               * self.goal_reward_weight)
            elif self._reward_type == 'sparse':
                goal_reward = int(goal_reached) * self.goal_reward_weight
        else:
            goal_reward = 0

        if self.velocity_reward_weight > 0:
            xy_velocities = self.get_body_comvel("torso")[:2]
            # rewards for speed on xy-plane (no matter which direction)
            velocity_reward = (self.velocity_reward_weight
                               * np.linalg.norm(xy_velocities))
        else:
            velocity_reward = 0

        if self.ctrl_cost_coeff > 0:
            lb, ub = self.action_bounds
            scaling = (ub - lb) * 0.5
            ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
                np.square(action / scaling))
        else:
            ctrl_cost = 0

        if self.contact_cost_coeff > 0:
            contact_cost = 0.5 * self.contact_cost_coeff * np.sum(
                np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        else:
            contact_cost = 0

        reward = (goal_reward + velocity_reward + self.survive_reward
                  - ctrl_cost - contact_cost)

        is_healthy = (np.isfinite(self._state).all()
                      and 0.2 <= self._state[2] <= 1.0)
        done = not is_healthy or (self.terminate_at_goal and goal_reached)

        next_observation = self.get_current_obs()
        info = {'goal_position': self.goal_position}
        return Step(next_observation, reward, done, **info)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        logs = get_random_goal_logs(
            paths,
            self.goal_radius,
            fixed_goal_position=getattr(self, 'fixed_goal_position', False))
        for row in logs:
            logger.record_tabular(*row)
