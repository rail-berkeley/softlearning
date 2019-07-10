import os.path as osp
import numpy as np

from gym.envs.mujoco.mujoco_env import MujocoEnv

from serializable import Serializable

from softlearning.misc.utils import PROJECT_PATH

"""
Environment based off of the SwimmerEnv found in rllab 
https://github.com/rll/rllab/blob/master/rllab/envs/mujoco/swimmer_env.py
"""
class RLLabSwimmerEnv(MujocoEnv, Serializable):

    MODEL_PATH = osp.abspath(
        osp.join(PROJECT_PATH, 'models', 'rllab_swimmer.xml'))
    ORI_IND = 2

    def __init__(
                self,
                ctrl_cost_coeff=1e-2,
                *args, **kwargs):
            self._Serializable__initialize(locals())
            self.ctrl_cost_coeff = ctrl_cost_coeff
            MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5, *args, **kwargs)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_ori(self):
        return self.sim.data.qpos[self.__class__.ORI_IND]

    def reset_model(self):
        self.set_state(self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01,
                self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1)
        return self._get_obs()

    def step(self, action):
        # pos_before = self.get_body_com("torso")[0]
        pos_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        pos_after = self.sim.data.qpos[0]
        next_obs = self._get_obs()
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = (pos_after - pos_before) / self.dt
        reward = forward_reward - ctrl_cost
        done = False
        env_infos = dict(reward_forward=forward_reward,
                         reward_ctrl=-ctrl_cost,
                        )
        
        return next_obs, reward, done, env_infos
