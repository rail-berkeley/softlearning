import os

import numpy as np

from gym import error, spaces
from gym.envs.mujoco import MujocoEnv as GymMujocoEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py,"
        " and also perform the setup instructions here:"
        " https://github.com/openai/mujoco-py/.)".format(e))


class CustomMujocoEnv(GymMujocoEnv):
    def __init__(self, model_path, frame_skip, automatically_set_spaces=False):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(
                os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        if automatically_set_spaces:
            observation, _reward, done, _info = self.step(
                np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high)

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)

        self.seed()
