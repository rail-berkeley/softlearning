import numpy as np

from rllab.core.serializable import Serializable
from rllab.mujoco_py import MjViewer
from rllab.misc import logger

from .pusher import PusherEnv


class ImagePusherEnv(PusherEnv):
    def __init__(self, image_size, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.image_size = image_size
        PusherEnv.__init__(self, *args, **kwargs)
        self.viewer_setup()

    def get_current_obs(self):
        self.viewer_setup()
        image = self.render(mode='rgb_array')

        return np.concatenate([
            image.reshape(-1),
            self.model.data.qpos.flat[self.JOINT_INDS],
            self.model.data.qvel.flat[self.JOINT_INDS],
        ]).reshape(-1)

    def step(self, action):
        """Step, computing reward from 'true' observations and not images."""

        reward_observations = super(ImagePusherEnv, self).get_current_obs()
        reward, info = self.compute_reward(reward_observations, action)

        self.forward_dynamics(action)
        observation = self.get_current_obs()
        done = False

        return observation, reward, done, info

    def get_viewer(self):
        if self.viewer is None:
            width, height = self.image_size[:2]
            self.viewer = MjViewer(
                visible=False, init_width=width, init_height=height)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def viewer_setup(self):
        viewer = self.get_viewer()
        viewer.cam.trackbodyid = 0
        cam_dist = 3.5
        viewer.cam.lookat[:3] = [0, 0, 0]
        viewer.cam.distance = cam_dist
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 0
        viewer.cam.trackbodyid = -1

    def render(self, *args, **kwargs):
        self.viewer_setup()
        return super(ImagePusherEnv, self).render(*args, **kwargs)


class ImageForkReacherEnv(ImagePusherEnv):
    def __init__(self,
                 arm_goal_distance_cost_coeff,
                 arm_object_distance_cost_coeff,
                 *args,
                 **kwargs):
        Serializable.quick_init(self, locals())

        self._arm_goal_distance_cost_coeff = arm_goal_distance_cost_coeff
        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff

        super(ImageForkReacherEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False

        arm_pos = observations[:, -6:-3]
        object_pos = observations[:, -3:]
        object_pos_masked = object_pos[:, :2][:, self._goal_mask]

        arm_goal_dists = np.linalg.norm(
            arm_pos[:, :2] - self._goal[None], axis=1)
        arm_object_dists = np.linalg.norm(
            arm_pos[:, :2] - object_pos_masked, axis=1)
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
            return super(PusherEnv, self).reset(init_state)

        qpos = np.random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos.squeeze()

        target_pos = np.random.uniform([-1.0], [1.0], size=[2])
        target_pos = np.sign(target_pos) * np.maximum(np.abs(target_pos), 1/3)
        target_pos[np.where(target_pos == 0)] = 1.0
        target_pos[1] += 1.0

        qpos[self.TARGET_INDS] = target_pos
        # qpos[self.TARGET_INDS] = [1.0, 2.0]
        # qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

        # TODO.before_release: Hack for reproducing the exact results we have in
        # paper, remove before release.
        puck_position = np.random.uniform(
            low=[0.3, -1.0], high=[1.0, -0.4]),

        qpos[self.PUCK_INDS] = puck_position

        qvel = self.init_qvel.copy().squeeze()
        qvel[self.PUCK_INDS] = 0
        qvel[self.TARGET_INDS] = 0

        qacc = np.zeros(self.model.data.qacc.shape[0])
        ctrl = np.zeros(self.model.data.ctrl.shape[0])

        full_state = np.concatenate((qpos, qvel, qacc, ctrl))
        super(PusherEnv, self).reset(full_state)

        return self.get_current_obs()

    def log_diagnostics(self, paths):
        arm_goal_dists = [
            p['env_infos'][-1]['arm_goal_distance']
            for p in paths]
        arm_object_dists = [
            p['env_infos'][-1]['arm_object_distance']
            for p in paths]

        logger.record_tabular('FinalArmDistanceAvg', np.mean(arm_goal_dists))
        logger.record_tabular('FinalArmDistanceMax', np.max(arm_goal_dists))
        logger.record_tabular('FinalArmDistanceMin', np.min(arm_goal_dists))
        logger.record_tabular('FinalArmDistanceStd', np.std(arm_goal_dists))

        logger.record_tabular('FinalGoalDistanceAvg', np.mean(arm_object_dists))
        logger.record_tabular('FinalGoalDistanceMax', np.max(arm_object_dists))
        logger.record_tabular('FinalGoalDistanceMin', np.min(arm_object_dists))
        logger.record_tabular('FinalGoalDistanceStd', np.std(arm_object_dists))
