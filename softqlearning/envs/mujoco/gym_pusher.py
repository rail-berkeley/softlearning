import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

from rllab.misc import logger


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, cylinder_pos=(0.0, 0.0), target_pos=(0.0, 0.0),
                 random_cylinder_pos=False,
                 tgt_cost_coeff=1.0, ctrl_cost_coeff=0.1, guide_cost_coeff=0.5):
        utils.EzPickle.__init__(self)

        self.tgt_cost_coeff = tgt_cost_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.guide_cost_coeff = guide_cost_coeff

        self.cylinder_pos = np.array(cylinder_pos)
        self.goal_pos = np.array(target_pos)
        self.random_cylinder_pos = random_cylinder_pos

        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def _step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = (self.tgt_cost_coeff * reward_dist
                  + self.ctrl_cost_coeff * reward_ctrl
                  + self.guide_cost_coeff * reward_near)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return (ob, reward, done,
                dict(reward_dist=reward_dist,
                     reward_ctrl=reward_ctrl,
                     reward_near=reward_near,
                     com_object=self.get_body_com('object'),
                     com_goal=self.get_body_com('goal')))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        while self.random_cylinder_pos:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    @staticmethod
    def log_diagnostics(paths):
        dist = []
        reward_dist = []
        reward_ctrl = []
        reward_near = []
        for path in paths:
            info = path['env_infos']
            dist.append(np.linalg.norm(
                info['com_object'][-1] - info['com_goal'][-1]
            ))
            reward_dist.append(info['reward_dist'])
            reward_ctrl.append(info['reward_ctrl'])
            reward_near.append(info['reward_near'])

        logger.record_tabular('env:goal_final_distance_mean', np.mean(dist))
        logger.record_tabular('env:goal_final_distance_std', np.std(dist))
        logger.record_tabular('env:reward_dist_mean', np.mean(reward_dist))
        logger.record_tabular('env:reward_dist_std', np.std(reward_dist))
        logger.record_tabular('env:reward_ctrl_mean', np.mean(reward_ctrl))
        logger.record_tabular('env:reward_ctrl_std', np.std(reward_ctrl))
        logger.record_tabular('env:reward_near_mean', np.mean(reward_near))
        logger.record_tabular('env:reward_near_std', np.std(reward_near))
