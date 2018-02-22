import numpy as np
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.misc import logger

from .helpers import get_multi_direction_logs

class MultiDirectionSwimmerEnv(SwimmerEnv):
    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        xy_velocities = self.get_body_comvel("torso")[:2]

        # rewards for speed on positive x direction
        xy_velocity = np.linalg.norm(xy_velocities)
        if xy_velocities[0] < 0:
            xy_velocity *= -1.0

        reward = xy_velocity - ctrl_cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        logs = get_multi_direction_logs(paths)
        for row in logs:
            logger.record_tabular(*row)
