import os.path as osp

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc.utils import PROJECT_PATH


class PusherEnv(MujocoEnv, Serializable):
    """Pusher environment

    Pusher is a two-dimensional 3-DoF manipulator. Task is to slide a cylinder-
    shaped object, or a 'puck', to a target coordinates.
    """
    FILE_PATH = osp.abspath(osp.join(PROJECT_PATH, 'models', 'pusher.xml'))

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    # TODO.before_release Fix target visualization (right now the target is
    # always drawn in (-1, 0), regardless of the actual goal.

    def __init__(self, goal=(0, -1), arm_distance_coeff=0):
        """
        goal (`list`): List of two elements denoting the x and y coordinates of
            the goal location. Either of the coordinate can also be a string
            'any' to make the reward not to depend on the corresponding
            coordinate.
        arm_distance_coeff ('float'): Coefficient for the arm-to-object distance
            cost.
        """
        super(PusherEnv, self).__init__(file_path=self.FILE_PATH)
        Serializable.quick_init(self, locals())

        self._goal_mask = [coordinate != 'any' for coordinate in goal]
        self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._arm_distance_coeff = arm_distance_coeff
        self._action_cost_coeff = 0.1

        # Make the the complete robot visible when visualizing.
        self.model.stat.extent = 10

    def step(self, action):
        reward, info = self.compute_reward(self.get_current_obs(), action)

        self.forward_dynamics(action)
        observation = self.get_current_obs()
        done = False

        return observation, reward, done, info

    def compute_reward(self, observations, actions):
        is_batch = False
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = True

        arm_pos = observations[:, -6:-3]
        obj_pos = observations[:, -3:]
        obj_pos_masked = obj_pos[:, :2][:, self._goal_mask]

        goal_dists = np.linalg.norm(self._goal[None] - obj_pos_masked, axis=1)
        arm_dists = np.linalg.norm(arm_pos - obj_pos, axis=1)
        ctrl_costs = np.sum(actions**2, axis=1)

        rewards = -self._action_cost_coeff * ctrl_costs - goal_dists
        rewards -= self._arm_distance_coeff * arm_dists

        if not is_batch:
            rewards = rewards.squeeze()
            arm_dists = arm_dists.squeeze()
            goal_dists = goal_dists.squeeze()

        return rewards, {
            'arm_distance': arm_dists,
            'goal_distance': goal_dists
        }

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=-0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset(self, init_state=None):
        if init_state:
            super(PusherEnv, self).reset(init_state)
            return

        qpos = np.random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()
        qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

        # TODO.before_release: Hack for reproducing the exact results we have in
        # paper, remove before release.
        while True:
            puck_position = np.random.uniform(
                low=[0.3, -1.0], high=[1.0, -0.4]),

            bottom_right_corner = np.array([1, -1])
            if np.linalg.norm(puck_position - bottom_right_corner) > 0.45:
                break

        qpos[self.PUCK_INDS] = puck_position

        qvel = self.init_qvel.copy().squeeze()
        qvel[self.PUCK_INDS] = 0
        qvel[self.TARGET_INDS] = 0

        qacc = np.zeros(self.model.data.qacc.shape[0])
        ctrl = np.zeros(self.model.data.ctrl.shape[0])

        full_state = np.concatenate((qpos, qvel, qacc, ctrl))
        super(PusherEnv, self).reset(full_state)

        return self.get_current_obs()

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[self.JOINT_INDS],
            self.model.data.qvel.flat[self.JOINT_INDS],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
        ]).reshape(-1)

    @overrides
    def log_diagnostics(self, paths):
        arm_dists = [p['env_infos'][-1]['arm_distance'] for p in paths]
        goal_dists = [p['env_infos'][-1]['goal_distance'] for p in paths]

        logger.record_tabular('FinalArmDistanceAvg', np.mean(arm_dists))
        logger.record_tabular('FinalArmDistanceMax', np.max(arm_dists))
        logger.record_tabular('FinalArmDistanceMin', np.min(arm_dists))
        logger.record_tabular('FinalArmDistanceStd', np.std(arm_dists))

        logger.record_tabular('FinalGoalDistanceAvg', np.mean(goal_dists))
        logger.record_tabular('FinalGoalDistanceMax', np.max(goal_dists))
        logger.record_tabular('FinalGoalDistanceMin', np.min(goal_dists))
        logger.record_tabular('FinalGoalDistanceStd', np.std(goal_dists))
