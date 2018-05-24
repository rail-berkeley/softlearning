import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.base import Step
from rllab.misc import logger

from .helpers import get_multi_direction_logs

class MultiDirectionBaseEnv(Serializable):
    def __init__(self,
                 velocity_reward_weight=1.0,
                 survive_reward=0,
                 ctrl_cost_coeff=0,
                 contact_cost_coeff=0,
                 velocity_deviation_cost_coeff=0,
                 *args, **kwargs):
        self._velocity_reward_weight = velocity_reward_weight
        self._survive_reward = survive_reward

        self._ctrl_cost_coeff = ctrl_cost_coeff
        self._contact_cost_coeff = contact_cost_coeff
        self._velocity_deviation_cost_coeff = velocity_deviation_cost_coeff
        Serializable.quick_init(self, locals())

    @property
    def velocity_reward(self):
        xy_velocities = self.get_body_comvel("torso")[:2]
        # rewards for speed on xy-plane (no matter which direction)
        xy_velocity = np.linalg.norm(xy_velocities)

        velocity_reward = self._velocity_reward_weight * xy_velocity
        return velocity_reward

    @property
    def survive_reward(self):
        return self._survive_reward

    def control_cost(self, action):
        lb, ub = self.action_bounds
        scaling = (ub - lb) / 2.0

        return 0.5 * self._ctrl_cost_coeff * np.sum(
            np.square(action / scaling))

    @property
    def contact_cost(self):
        return 0.5 * self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),

    @property
    def is_healthy(self):
        return True

    @property
    def velocity_deviation_cost(self):
        velocity_deviation_cost = (
            0.5 *
            self._velocity_deviation_cost_coeff
            * np.sum(np.square(self.get_body_comvel("torso")[2:])))
        return velocity_deviation_cost

    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):
        self.forward_dynamics(action)

        reward = (
            self.velocity_reward
            + self.survive_reward
            - self.control_cost(action)
            - self.contact_cost
            - self.velocity_deviation_cost)

        next_observation = self.get_current_obs()
        return Step(next_observation, float(reward), self.done)

    def log_diagnostics(self, paths, *args, **kwargs):
        logs = get_multi_direction_logs(paths)
        for row in logs:
            logger.record_tabular(*row)


class MultiDirectionSwimmerEnv(MultiDirectionBaseEnv, SwimmerEnv):
    def __init__(self,
                 ctrl_cost_coeff=1e-2,
                 *args, **kwargs):
        MultiDirectionBaseEnv.__init__(
            self, ctrl_cost_coeff=ctrl_cost_coeff, *args, **kwargs)
        SwimmerEnv.__init__(
            self, ctrl_cost_coeff=ctrl_cost_coeff, *args, **kwargs)

    @property
    def velocity_reward(self):
        xy_velocities = self.get_body_comvel("torso")[:2]

        # rewards for speed on positive x direction
        xy_velocity = np.linalg.norm(xy_velocities)
        if xy_velocities[0] < 0:
            xy_velocity *= -1.0

        velocity_reward = self._velocity_reward_weight * xy_velocity
        return velocity_reward

class MultiDirectionAntEnv(MultiDirectionBaseEnv, AntEnv):
    def __init__(self,
                 ctrl_cost_coeff=1e-2,
                 contact_cost_coeff=1e-3,
                 survive_reward=5e-2,
                 *args, **kwargs):
        MultiDirectionBaseEnv.__init__(
            self,
            ctrl_cost_coeff=ctrl_cost_coeff,
            contact_cost_coeff=contact_cost_coeff,
            survive_reward=survive_reward,
            *args, **kwargs)
        AntEnv.__init__(self,  *args, **kwargs)

    @property
    def is_healthy(self):
        return (np.isfinite(self._state).all()
                and 0.2 <= self._state[2] <= 1.0)

class MultiDirectionHumanoidEnv(MultiDirectionBaseEnv, HumanoidEnv):
    def __init__(self,
                 survive_reward=2e-1,
                 ctrl_cost_coeff=1e-3,
                 contact_cost_coeff=1e-5,
                 velocity_deviation_cost_coeff=1e-2,
                 *args, **kwargs):
        MultiDirectionBaseEnv.__init__(
            self,
            survive_reward=survive_reward,
            ctrl_cost_coeff=ctrl_cost_coeff,
            contact_cost_coeff=contact_cost_coeff,
            velocity_deviation_cost_coeff=velocity_deviation_cost_coeff,
            *args, **kwargs)
        HumanoidEnv.__init__(
            self,
            # survive_reward=survive_reward,
            alive_bonus=survive_reward, # TODO: remove this
            ctrl_cost_coeff=ctrl_cost_coeff,
            # contact_cost_coeff=contact_cost_coeff,
            impact_cost_coeff=contact_cost_coeff, # TODO: remove this
            vel_deviation_cost_coeff=velocity_deviation_cost_coeff, # TODO: remove this
            *args, **kwargs)

    @property
    def is_healthy(self):
        return 0.8 < self.model.data.qpos[2] < 2.0
