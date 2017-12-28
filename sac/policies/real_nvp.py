"""Real NVP policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from sac.distributions import RealNVPBijector

class RealNVPPolicy(object):
    """Real NVP policy"""

    def __init__(self, env_spec, real_nvp_config=None, squash=True, qf=None):
        """Initialize Real NVP policy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            real_nvp_config (`dict`): Parameter
                configuration for real nvp distribution.
            squash (`bool`): If True, squash the action samples between
                -1 and 1 with tanh.
            qf (`ValueFunction`): Q-function approximator.
        """
        self.real_nvp_config = real_nvp_config

        self._env_spec = env_spec
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._qf = qf

        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        ds = tf.contrib.distributions
        self.bijector = RealNVPBijector(config=real_nvp_config,
                                        event_ndims=self._Ds)
        self.base_distribution = ds.MultivariateNormalDiag(
            loc=tf.zeros(self._Ds), scale_diag=tf.ones(self._Ds))

        self.distribution = ds.TransformedDistribution(
            distribution=self.base_distribution,
            bijector=self.bijector,
            name="RealNVPPolicyDistribution")


        y = self.distribution.bijector.forward(self._observations_ph)
        self.log_pi = self.distribution.log_prob(y)
        self.pi = tf.exp(self.log_pi)
        self._action = tf.tanh(self.pi) if squash else self.pi

    def get_action(self, observations):
        """Sample action based on the observations.

        TODO: implement
        """
        return super().get_action(observations[None])[0], None

    def get_actions(self, observations):
        feed_dict = {self._observations_ph: observations}
        actions = tf.get_default_session().run(
            self._action, feed_dict=feed_dict)
        return actions

    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        TODO: implement
        """
        pass
