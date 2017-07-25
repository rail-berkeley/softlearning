import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides

from softqlearning.envs.mujoco.gym_hopper import HopperEnv as GymHopperEnv
from softqlearning.envs.gym_env import GymEnv

from softqlearning.misc import logger

class DummySpec:
    id = 0
    timestep_limit = 100


class HopperEnv(GymEnv, Parameterized):
    def __init__(self, **kwargs):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        env = GymHopperEnv(**kwargs)
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

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][0] - path["observations"][0][0]
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    @overrides
    def get_params_internal(self, **tags):
        return list()
