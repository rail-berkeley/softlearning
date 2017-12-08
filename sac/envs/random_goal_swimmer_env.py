"""Implements a swimmer which is sparsely rewarded for reaching a goal"""

import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_GOAL_RADIUS = 0.25

class RandomGoalSwimmerEnv(SwimmerEnv):
    """Implements a swimmer which is sparsely rewarded for reaching a goal"""

    @overrides
    def __init__(self,
                 goal_reward=10,
                 goal_radius=DEFAULT_GOAL_RADIUS,
                 *args,
                 **kwargs):
        self.goal_reward = goal_reward
        self.goal_radius = goal_radius
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def reset(self, goal_position=None, *args, **kwargs):
        if goal_position is None:
            goal_position = np.random.uniform(low=-5.0, high=5.0, size=(2,))

        self.goal_position = goal_position

        return super().reset(*args, **kwargs)

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        xy_position = self.model.data.xpos[0, :2]
        goal_distance = np.sqrt(np.sum((xy_position - self.goal_position)**2))

        # TODO: control cost?
        if goal_distance < self.goal_radius:
            reward, done = self.goal_reward, True
        else:
            reward, done = 0.0, False

        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        """Log diagnostic information based on past paths

        TODO: figure out what this should log and implement
        """
        super().log_diagnostics(paths, *args, **kwargs)
