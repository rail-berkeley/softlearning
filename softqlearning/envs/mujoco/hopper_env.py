import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import autoargs
from softqlearning.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.envs.mujoco.mujoco_env import MujocoEnv


# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)

import os.path as osp

MODEL_ROUGH = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        '../../../assets/hopper_rough.xml'
    )
)


class HopperEnv(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    @autoargs.arg('alive_coeff', type=float,
                  help='reward coefficient for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            alive_coeff=0.5,
            ctrl_cost_coeff=0.01,
            rough_terrain=False,
            multidirection=False,
            speed_threshold=False,  #Fixed reward for higher velocities, 0 o.w.
            *args, **kwargs):

        self.multidirection = multidirection
        self.speed_threshold = speed_threshold
        self.alive_coeff = alive_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        if rough_terrain:
            super(HopperEnv, self).__init__(*args,
                                            file_path=MODEL_ROUGH, **kwargs)
        else:
            super(HopperEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1].flat,
            self.model.data.qpos[2:].flat,
            np.clip(self.model.data.qvel, -10, 10).flat,
            np.clip(self.model.data.qfrc_constraint, -10, 10).flat,
            #self.get_body_com("torso").flat,
        ])

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = self.get_body_comvel("torso")[0]
        if self.multidirection:
            vel = np.abs(vel)

        if self.speed_threshold:
            vel_reward = 1 if vel > 0.2 else 0
        else:
            vel_reward = vel

        reward = vel_reward + self.alive_coeff - \
                0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        state = self._state
        #notdone = np.isfinite(state).all() and \
        #    (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
        #    (abs(state[2]) < .2)
        notdone = np.isfinite(state).all() and \
                  (np.abs(state[3:]) < 100).all() and (state[0] > .7)
        done = not notdone

        #done = False
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        return Step(next_obs, reward, done, com=com)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    def log_stats(self, paths):
        # forward distance
        progs = []
        for path in paths:
            coms = path["env_infos"]["com"]
            progs.append(coms[-1][0] - coms[0][0])
            # x-coord of com at the last time step minus the 1st step

        stats = {
            'env: ForwardProgressAverage': np.mean(progs),
            'env: ForwardProgressMax': np.max(progs),
            'env: ForwardProgressMin': np.min(progs),
            'env: ForwardProgressStd': np.std(progs),
            'env: ForwardProgressDiff': np.max(progs) - np.min(progs),
        }

        # HopperEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            com = path['env_infos']['com']
            xx = com[:, 0]
            zz = com[:, 2]
            ax.plot(xx, zz, 'b')
        ax.set_xlim((-1, np.max(xx)+1))
        ax.set_ylim((-1, 2))
