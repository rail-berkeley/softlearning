import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml


INIT_POS = np.array([0.2,0.15])
TARGET = np.array([0.2, -0.15]) + INIT_POS
DIST_THRESH = 0.05

class TwoDMaze(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, get_asset_xml('twod_maze.xml'), 2)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        dist = np.linalg.norm(pos - TARGET)
        dist_cost = 1 if dist>DIST_THRESH else dist/DIST_THRESH
        reward = - (dist_cost)# + 1e-2*np.linalg.norm(a))
        #print(reward, dist)
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        #return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()
        return np.concatenate([self.model.data.qpos]).ravel()

    def viewer_setup(self):
        v = self.viewer
        #v.cam.trackbodyid=0
        #v.cam.distance = v.model.stat.extent
