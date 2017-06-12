import numpy as np

from gym.core import Wrapper
from rllab.envs.gym_env import GymEnv


def get_base_env(obj):
    while isinstance(obj, Wrapper):
            obj = obj.env

    return obj


class PusherEnv(GymEnv):
    def __init__(self):
        super().__init__(
            env_name='Pusher-v0',
            record_video=False,
            video_schedule=None,
            log_dir=None,
            record_log=False,
            force_reset=False,
        )

        self._base_env = get_base_env(self.env)

        self.cylinder_pos = np.array([-0.3, 0])
        self.reset()

    def reset(self):
        obs = super().reset()
        #obs = self._base_env._reset()

        # Move cylinder back to fixed location.
        self._base_env.init_qpos[-4:-2] = self.cylinder_pos

        return obs



# pp = PusherEnv()
# pp.reset()
# pp.render()
# import ipdb; ipdb.set_trace()
