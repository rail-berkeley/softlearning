import os.path as osp

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softlearning.misc.utils import PROJECT_PATH


class Pusher2dEnv(Serializable, MujocoEnv):
    """Pusher environment

    Pusher is a two-dimensional 3-DoF manipulator. Task is to slide a cylinder-
    shaped object, or a 'puck', to a target coordinates.

    Note: Serializable has to be the first super class for classes extending
    MujocoEnv (or at least occur before MujocoEnv). Otherwise MujocoEnv calls
    Serializable.__init__ (from MujocoEnv.__init__), and the Serializable
    attributes (_Serializable__args and _Serializable__kwargs) will get
    overwritten.
    """
    FILE_PATH = osp.abspath(osp.join(PROJECT_PATH, 'models', 'pusher_2d.xml'))

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    # TODO.before_release Fix target visualization (right now the target is
    # always drawn in (-1, 0), regardless of the actual goal.

    def __init__(self,
                 goal=(0, -1),
                 arm_object_distance_cost_coeff=0,
                 goal_object_distance_cost_coeff=1.0,
                 action_cost_coeff=0.1):
        """
        goal (`list`): List of two elements denoting the x and y coordinates of
            the goal location. Either of the coordinate can also be a string
            'any' to make the reward not to depend on the corresponding
            coordinate.
        arm_object_distance_cost_coeff ('float'): Coefficient for the
            arm-to-object distance cost.
        goal_object_distance_cost_coeff ('float'): Coefficient for the
            goal-to-object distance cost.
        """
        Serializable.quick_init(self, locals())
        MujocoEnv.__init__(self, file_path=self.FILE_PATH)

        self._goal_mask = [coordinate != 'any' for coordinate in goal]
        self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff
        self._goal_object_distance_cost_coeff = goal_object_distance_cost_coeff
        self._action_cost_coeff = action_cost_coeff

        # Make the the complete robot visible when visualizing.
        self.model.stat.extent = 10

    def step(self, action):
        reward, info = self.compute_reward(self.get_current_obs(), action)

        self.forward_dynamics(action)
        observation = self.get_current_obs()
        done = False

        return observation, reward, done, info

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False

        arm_pos = observations[:, -6:-3]
        obj_pos = observations[:, -3:]
        obj_pos_masked = obj_pos[:, :2][:, self._goal_mask]

        goal_object_distances = np.linalg.norm(
            self._goal[None] - obj_pos_masked, axis=1)
        arm_object_dists = np.linalg.norm(arm_pos - obj_pos, axis=1)
        ctrl_costs = np.sum(actions**2, axis=1)

        costs = (
            + self._arm_object_distance_cost_coeff * arm_object_dists
            + self._goal_object_distance_cost_coeff * goal_object_distances
            + self._action_cost_coeff * ctrl_costs)

        rewards = -costs

        if not is_batch:
            rewards = rewards.squeeze()
            arm_object_dists = arm_object_dists.squeeze()
            goal_object_distances = goal_object_distances.squeeze()

        return rewards, {
            'arm_object_distance': arm_object_dists,
            'goal_object_distance': goal_object_distances
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
            return super(Pusher2dEnv, self).reset(init_state)

        qpos = np.random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()
        qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

        # TODO.before_release: Hack for reproducing the exact results we have
        # in paper, remove before release.
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
        super(Pusher2dEnv, self).reset(full_state)

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
        arm_object_distances = [
            p['env_infos'][-1]['arm_object_distance'] for p in paths]
        goal_object_distances = [
            p['env_infos'][-1]['goal_object_distance'] for p in paths]

        logger.record_tabular(
            'FinalArmObjectDistanceAvg', np.mean(arm_object_distances))
        logger.record_tabular(
            'FinalArmObjectDistanceMax', np.max(arm_object_distances))
        logger.record_tabular(
            'FinalArmObjectDistanceMin', np.min(arm_object_distances))
        logger.record_tabular(
            'FinalArmObjectDistanceStd', np.std(arm_object_distances))

        logger.record_tabular(
            'FinalGoalObjectDistanceAvg', np.mean(goal_object_distances))
        logger.record_tabular(
            'FinalGoalObjectDistanceMax', np.max(goal_object_distances))
        logger.record_tabular(
            'FinalGoalObjectDistanceMin', np.min(goal_object_distances))
        logger.record_tabular(
            'FinalGoalObjectDistanceStd', np.std(goal_object_distances))


class ForkReacherEnv(Pusher2dEnv):
    def __init__(self,
                 arm_goal_distance_cost_coeff,
                 arm_object_distance_cost_coeff,
                 *args,
                 **kwargs):
        Serializable.quick_init(self, locals())
        self._arm_goal_distance_cost_coeff = arm_goal_distance_cost_coeff
        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff

        super(ForkReacherEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False
        else:
            raise NotImplementedError('Might be broken.')

        arm_pos = observations[:, -8:-6]
        goal_pos = observations[:, -2:]
        object_pos = observations[:, -5:-3]

        arm_goal_dists = np.linalg.norm(arm_pos - goal_pos, axis=1)
        arm_object_dists = np.linalg.norm(arm_pos - object_pos, axis=1)
        ctrl_costs = np.sum(actions**2, axis=1)

        costs = (
            + self._arm_goal_distance_cost_coeff * arm_goal_dists
            + self._arm_object_distance_cost_coeff * arm_object_dists
            + self._action_cost_coeff * ctrl_costs)

        rewards = -costs

        if not is_batch:
            rewards = rewards.squeeze()
            arm_goal_dists = arm_goal_dists.squeeze()
            arm_object_dists = arm_object_dists.squeeze()

        return rewards, {
            'arm_goal_distance': arm_goal_dists,
            'arm_object_distance': arm_object_dists,
        }

    def reset(self, init_state=None):
        if init_state:
            return super(Pusher2dEnv, self).reset(init_state)

        qpos = np.random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()

        qpos[self.JOINT_INDS[0]] = np.random.uniform(-np.pi, np.pi)
        qpos[self.JOINT_INDS[1]] = np.random.uniform(
            -np.pi/2, np.pi/2) + np.pi/4
        qpos[self.JOINT_INDS[2]] = np.random.uniform(
            -np.pi/2, np.pi/2) + np.pi/2

        target_pos = np.random.uniform([-1.0], [1.0], size=[2])
        target_pos = np.sign(target_pos) * np.maximum(np.abs(target_pos), 1/2)
        target_pos[np.where(target_pos == 0)] = 1.0
        target_pos[1] += 1.0

        qpos[self.TARGET_INDS] = target_pos
        # qpos[self.TARGET_INDS] = [1.0, 2.0]
        # qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

        puck_position = np.random.uniform([-1.0], [1.0], size=[2])
        puck_position = (
            np.sign(puck_position)
            * np.maximum(np.abs(puck_position), 1/2))
        puck_position[np.where(puck_position == 0)] = 1.0
        # puck_position[1] += 1.0
        # puck_position = np.random.uniform(
        #     low=[0.3, -1.0], high=[1.0, -0.4]),

        qpos[self.PUCK_INDS] = puck_position

        qvel = self.init_qvel.copy().squeeze()
        qvel[self.PUCK_INDS] = 0
        qvel[self.TARGET_INDS] = 0

        qacc = np.zeros(self.model.data.qacc.shape[0])
        ctrl = np.zeros(self.model.data.ctrl.shape[0])

        full_state = np.concatenate((qpos, qvel, qacc, ctrl))
        super(Pusher2dEnv, self).reset(full_state)

        return self.get_current_obs()

    def log_diagnostics(self, paths):
        arm_goal_dists = [
            p['env_infos'][-1]['arm_goal_distance']
            for p in paths]
        arm_object_dists = [
            p['env_infos'][-1]['arm_object_distance']
            for p in paths]

        logger.record_tabular(
            'FinalArmGoalDistanceAvg', np.mean(arm_goal_dists))
        logger.record_tabular(
            'FinalArmGoalDistanceMax', np.max(arm_goal_dists))
        logger.record_tabular(
            'FinalArmGoalDistanceMin', np.min(arm_goal_dists))
        logger.record_tabular(
            'FinalArmGoalDistanceStd', np.std(arm_goal_dists))

        logger.record_tabular(
            'FinalArmObjectDistanceAvg', np.mean(arm_object_dists))
        logger.record_tabular(
            'FinalArmObjectDistanceMax', np.max(arm_object_dists))
        logger.record_tabular(
            'FinalArmObjectDistanceMin', np.min(arm_object_dists))
        logger.record_tabular(
            'FinalArmObjectDistanceStd', np.std(arm_object_dists))

    def get_current_obs(self):
        super_observation = super(ForkReacherEnv, self).get_current_obs()
        observation = np.concatenate([
            super_observation, self.get_body_com('goal')[:2]
        ])
        return observation
