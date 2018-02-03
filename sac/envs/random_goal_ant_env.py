"""Implements a ant which is sparsely rewarded for reaching a goal"""

import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger, autoargs

REWARD_TYPES = ('dense', 'sparse')

class RandomGoalAntEnv(AntEnv):
    """Implements a ant env which is sparsely rewarded for reaching a goal"""

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(self,
                 reward_type='dense',
                 goal_reward_weight=1e-3,
                 goal_radius=0.25,
                 ctrl_cost_coeff=1e-2,
                 contact_cost_coeff=1e-3,
                 survive_reward = 5e-2,
                 terminate_at_goal=True,
                 *args,
                 **kwargs):
        assert reward_type in REWARD_TYPES

        self._reward_type = reward_type
        self.goal_reward_weight = goal_reward_weight
        self.goal_radius = goal_radius
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.contact_cost_coeff = contact_cost_coeff
        self.survive_reward = survive_reward
        self.terminate_at_goal = terminate_at_goal
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def reset(self, goal_position=None, *args, **kwargs):
        if goal_position is None:
            goal_position_x = 5.0
            goal_position_y = np.random.uniform(low=-5.0, high=5.0)
            goal_position = np.array([goal_position_x, goal_position_y])

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

        xy_position = self.get_body_com('torso')[:2]
        self.goal_distance = np.linalg.norm(xy_position - self.goal_position)

        goal_reached = self.goal_distance < self.goal_radius

        if self._reward_type == 'dense':
            goal_reward = -self.goal_distance * self.goal_reward_weight
        elif self._reward_type == 'sparse':
            goal_reward = int(goal_reached) * self.goal_reward_weight

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

        reward = goal_reward + self.survive_reward - ctrl_cost - contact_cost

        is_healthy = (np.isfinite(self._state).all()
                      and 0.2 <= self._state[2] <= 1.0)
        done = not is_healthy or (self.terminate_at_goal and goal_reached)

        next_observation = self.get_current_obs()
        info = {'goal_position': self.goal_position}
        return Step(next_observation, reward, done, **info)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        """Log diagnostic information based on past paths

        TODO: figure out what this should log and implement
        """
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-5] - path["observations"][0][-5]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))

        logger.record_tabular('FinalDistanceFromGoal', self.goal_distance)
        origin_distance_from_goal = np.sqrt(np.sum(self.goal_position**2))
        logger.record_tabular('OriginDistanceFromGoal',
                              origin_distance_from_goal)
        progress_towards_goal = origin_distance_from_goal - self.goal_distance
        logger.record_tabular('ProgressTowardsGoal', progress_towards_goal)
