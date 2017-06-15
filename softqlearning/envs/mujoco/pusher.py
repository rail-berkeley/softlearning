from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides

from softqlearning.envs.mujoco.gym_pusher import PusherEnv as GymPusherEnv
from softqlearning.envs.gym_env import GymEnv


class DummySpec:
    id = 0
    timestep_limit = 100


class PusherEnv(GymEnv, Parameterized):
    def __init__(self, **kwargs):
        Parameterized.__init__(self)
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

    def log_diagnostics(self, paths):
        return self.env.log_diagnostics(paths)

    @overrides
    def get_params_internal(self, **tags):
        return list()
