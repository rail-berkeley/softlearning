import os.path as osp
import numpy as np

from gym.envs.mujoco.mujoco_env import MujocoEnv

from serializable import Serializable

from softlearning.misc.utils import PROJECT_PATH

"""
Environment based off of the HumanoidEnv found in rllab 
https://github.com/rll/rllab/blob/master/rllab/envs/mujoco/simple_humanoid_env.py
"""
class RLLabHumanoidEnv(Serializable, MujocoEnv):

    MODEL_PATH = osp.abspath(
        osp.join(PROJECT_PATH, 'models', 'rllab_humanoid.xml'))

    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            *args, **kwargs):
        self._Serializable__initialize(locals())
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5, *args, **kwargs)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            np.clip(data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso").flat,
        ])

    def _get_com(self):
        data = self.sim.data
        mass = self.model.body_mass[:, None]
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))


    def step(self, action):
        pos_before = self._get_com()
        self.do_simulation(action, self.frame_skip)
        pos_after = self._get_com()
        next_obs = self._get_obs()

        alive_bonus = self.alive_bonus
        data = self.sim.data

        # velocity computation was originally done by mujoco rather than finite differences
        comvel = (pos_after - pos_before) / self.dt
        lin_vel_reward = comvel[0]
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost
        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0
        env_infos = dict(reward_linvel=lin_vel_reward,
                         reward_ctrl=-ctrl_cost,
                         reward_alive=alive_bonus)

        return next_obs, reward, done, env_infos
    
    def reset_model(self):
        self.set_state(self.init_qpos + np.random.normal(size=self.init_qpos.shape) * 0.01,
                self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1)
        return self._get_obs()

