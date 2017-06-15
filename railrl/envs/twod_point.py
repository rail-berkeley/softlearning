import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from railrl.envs.env_utils import get_asset_xml


TARGET = np.array([0.2, 0])

class TwoDPoint(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, get_asset_xml('twod_point.xml'), 2)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        dist = np.linalg.norm(pos - TARGET)
        reward = - (dist + 1e-2*np.linalg.norm(a))
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

def make_heat_map(eval_func, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution)
    map = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            map[i,j] = eval_func(np.array([linspace[i], linspace[j]]))
    return map

def make_density_map(paths, resolution=50):
    linspace = np.linspace(-0.3, 0.3, num=resolution+1)
    y = paths[:,0]
    x = paths[:,1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(linspace, linspace))
    H = H.astype(np.float)
    H = H/np.max(H)
    return H

def plot_maps(old_combined=None, *heatmaps):
    import matplotlib.pyplot as plt
    combined = np.c_[heatmaps]
    if old_combined is not None:
        combined = np.r_[old_combined, combined]
    plt.figure()
    plt.imshow(combined, cmap='afmhot', interpolation='none')
    plt.show()
    return combined

if __name__ == "__main__":
    def evalfn(a):
        return np.linalg.norm(a)
    hm = make_heat_map(evalfn, resolution=50)
    paths = np.random.randn(5000,2)*0.1
    dm = make_density_map(paths, resolution=50)
    a = plot_maps(None, hm, dm)
    plot_maps(a, hm, dm)
