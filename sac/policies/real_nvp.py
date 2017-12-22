"""Real NVP policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sac.distributions import RealNVP
from sac.policies import NNPolicy


class RealNVPPolicy(NNPolicy, Serializable):
    """Real NVP policy"""

    def __init__(self, env_spec, real_nvp_config, squash=True, qf=None):
        """Initialize Real NVP policy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            real_nvp_config (`sac.distributions.real_nvp.Config`): Parameter
                configuration for real nvp distribution.
            squash (`bool`): If True, squash the action samples between
                -1 and 1 with tanh.
            qf (`ValueFunction`): Q-function approximator.
        """
        Serializable.quick_init(self, locals())

        self.real_nvp_config = real_nvp_config
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._qf = qf

        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._Ds],
            name='observations',
        )

        self.distribution = RealNVP(real_nvp_config)

        super().__init__(
            env_spec,
            obs_pl=self._observations_ph,
            action=(
                tf.tanh(self._distribution_z)
                if squash
                else self._distribution_z
            ),
            scope_name='policy'
        )

    @overrides
    def get_action(self, observations):
        """Sample action based on the observations.

        TODO: implement
        """
        return super().get_action(observations)

    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        TODO: implement
        """
        pass
