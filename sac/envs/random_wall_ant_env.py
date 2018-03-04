"""Implements a multi-direction ant with blocking walls"""

import os

import numpy as np

from sac.misc.utils import PROJECT_PATH
from .multi_direction_ant_env import MultiDirectionAntEnv

MODELS_PATH = os.path.abspath(
    os.path.join(PROJECT_PATH, 'sac/mujoco_models'))

class RandomWallAntEnv(MultiDirectionAntEnv):
    """Implements a multi-direction ant with blocking walls"""

    FILE_PATH = os.path.join(MODELS_PATH, 'random_wall_ant.xml')

    def __init__(self, *args, **kwargs):
        file_path = self.__class__.FILE_PATH
        kwargs.pop('file_path', None)
        super(RandomWallAntEnv, self).__init__(
            file_path=file_path, *args, **kwargs)

    def reset(self, *args, **kwargs):
        observation = super().reset(*args, **kwargs)

        wall_names = ['wall-{}'.format(i) for i in range(4)]
        wall_idx = [self.model.geom_names.index(wall_name) for wall_name in wall_names]
        hide_idx = np.random.choice(wall_idx, 2, replace=False)
        wall_positions = [
            (0, 5, 0.5),
            (5, 0, 0.5),
            (0, -5, 0.5),
            (-5, 0, 0.5),
        ]

        new_geom_pos = self.model.geom_pos.copy()
        new_geom_pos[wall_idx, ...] = wall_positions
        new_geom_pos[hide_idx, 2] = 20 # raise the wall in the air
        self.model.geom_pos = new_geom_pos

        return observation
