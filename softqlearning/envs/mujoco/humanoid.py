import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides

from softqlearning.envs.mujoco.gym_humanoid import HumanoidEnv as GymHumanoid
from softqlearning.envs.gym_env import GymEnv

from softqlearning.misc import logger


class DummySpec:
    id = 0
    timestep_limit = 100


class HumanoidEnv(GymEnv, Parameterized):
    def __init__(self, record_video=False, record_log=False, **kwargs):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        env = GymHumanoid(**kwargs)
        env.spec = DummySpec()

        super().__init__(
            env=env,
            record_video=record_video,
            video_schedule=None,
            log_dir=None,
            record_log=record_log,
            force_reset=False,
        )

        self.reset()

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path['env_infos']['comx'][-1] - path['env_infos']['comx'][0]
            for path in paths
        ]
        logger.record_tabular('ForwardProgressAvg', np.mean(progs))
        logger.record_tabular('ForwardProgressMax', np.max(progs))
        logger.record_tabular('ForwardProgressMin', np.min(progs))
        logger.record_tabular('ForwardProgressStd', np.std(progs))

    @overrides
    def get_params_internal(self, **tags):
        return list()
