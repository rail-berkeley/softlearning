import numpy as np
from gym import utils

from softlearning.environments.helpers import random_point_in_circle
from .pusher_2d import Pusher2dEnv


class ImagePusher2dEnv(Pusher2dEnv):
    def __init__(self, image_shape, *args, **kwargs):
        utils.EzPickle.__init__(**locals())
        self.image_shape = image_shape
        Pusher2dEnv.__init__(self, *args, **kwargs)

    def _get_obs(self):
        width, height = self.image_shape[:2]
        image = self.render(mode='rgb_array', width=width, height=height)
        image = ((2.0 / 255.0) * image - 1.0)

        return np.concatenate([
            image.reshape(-1),
            self.sim.data.qpos.flat[self.JOINT_INDS],
            self.sim.data.qvel.flat[self.JOINT_INDS],
        ]).reshape(-1)

    def step(self, action):
        """Step, computing reward from 'true' observations and not images."""

        reward_observations = super(ImagePusher2dEnv, self)._get_obs()
        reward, info = self.compute_reward(reward_observations, action)

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        done = False

        return observation, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[:3] = [0, 0, 0]
        self.viewer.cam.distance = 3.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
        self.viewer.cam.trackbodyid = -1


class ImageForkReacher2dEnv(ImagePusher2dEnv):
    def __init__(self,
                 arm_goal_distance_cost_coeff,
                 arm_object_distance_cost_coeff,
                 *args,
                 **kwargs):
        utils.EzPickle.__init__(**locals())

        self._arm_goal_distance_cost_coeff = arm_goal_distance_cost_coeff
        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff

        super(ImageForkReacher2dEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False
        else:
            raise NotImplementedError('Might be broken.')

        arm_pos = observations[:, -6:-4]
        goal_pos = self.get_body_com('goal')[:2][None]
        object_pos = observations[:, -3:-1]

        arm_goal_dists = np.linalg.norm(arm_pos - goal_pos, axis=1)
        arm_object_dists = np.linalg.norm(arm_pos - object_pos, axis=1)
        ctrl_costs = np.sum(actions**2, axis=1)

        costs = (
            + self._arm_goal_distance_cost_coeff * arm_goal_dists
            + self._arm_object_distance_cost_coeff * arm_object_dists
            + self._ctrl_cost_coeff * ctrl_costs)

        rewards = -costs

        if not is_batch:
            rewards = rewards.squeeze()
            arm_goal_dists = arm_goal_dists.squeeze()
            arm_object_dists = arm_object_dists.squeeze()

        return rewards, {
            'arm_goal_distance': arm_goal_dists,
            'arm_object_distance': arm_object_dists,
        }

    def reset_model(self):
        qpos = np.random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()

        # qpos[self.JOINT_INDS[0]] = np.random.uniform(-np.pi, np.pi)
        # qpos[self.JOINT_INDS[1]] = np.random.uniform(
        #     -np.pi/2, np.pi/2) + np.pi/4
        # qpos[self.JOINT_INDS[2]] = np.random.uniform(
        #     -np.pi/2, np.pi/2) + np.pi/2

        target_position = np.array(random_point_in_circle(
            angle_range=(0, 2*np.pi), radius=(0.6, 1.2)))
        target_position[1] += 1.0

        qpos[self.TARGET_INDS] = target_position
        # qpos[self.TARGET_INDS] = [1.0, 2.0]
        # qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

        puck_position = np.random.uniform([-1.0], [1.0], size=[2])
        puck_position = (
            np.sign(puck_position)
            * np.maximum(np.abs(puck_position), 1/2))
        puck_position[np.flatnonzero(puck_position == 0)] = 1.0
        # puck_position[1] += 1.0
        # puck_position = np.random.uniform(
        #     low=[0.3, -1.0], high=[1.0, -0.4]),

        qpos[self.PUCK_INDS] = puck_position

        qvel = self.init_qvel.copy().squeeze()
        qvel[self.PUCK_INDS] = 0
        qvel[self.TARGET_INDS] = 0

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        # super(Pusher2dEnv, self).reset(full_state)

        self.set_state(qpos, qvel)

        return self._get_obs()


class BlindForkReacher2dEnv(ImageForkReacher2dEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[self.JOINT_INDS],
            self.sim.data.qvel.flat[self.JOINT_INDS],
        ]).reshape(-1)
