from rllab.envs.gym_env import GymEnv
from rllab.core.serializable import Serializable
from softqlearning.envs.mujoco.gym_pusher import PusherEnv as GymPusherEnv


class DummySpec:
    id = 0
    timestep_limit = 100


class PusherEnv(GymEnv):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        env = GymPusherEnv(**kwargs)
        env.spec = DummySpec()

        super().__init__(
            env=env,
            record_video=False,
            video_schedule=None,
            log_dir=None,
            record_log=False,
            force_reset=False,
        )

        self.reset()
