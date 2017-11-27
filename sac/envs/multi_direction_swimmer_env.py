import numpy as np
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

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
        # rewards for speed on xy-plane (no matter which direction)
        xy_velocity = np.sqrt(np.sum(xy_velocities**2))
        reward = xy_velocity - ctrl_cost
        done = False
        return Step(next_obs, reward, done)
