"""Real NVP policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from sac.distributions import RealNVPBijector

DEFAULT_CONFIG = {
    "mode": "train",
    "D_in": 2,
    "learning_rate": 1e-4,
    "squash": False,
    "real_nvp_config": None
}

class RealNVPPolicy(object):
    """Real NVP policy"""

    def __init__(self, env_spec, config=None, qf=None):
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
        self.config = dict(DEFAULT_CONFIG, **(config or {}))

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
        self.bijector = RealNVPBijector(
            config=self.config["real_nvp_config"], event_ndims=self._Ds)

        self.base_distribution = ds.MultivariateNormalDiag(
            loc=tf.zeros(self._Ds), scale_diag=tf.ones(self._Ds))

        self.distribution = ds.TransformedDistribution(
            distribution=self.base_distribution,
            bijector=self.bijector,
            name="RealNVPPolicyDistribution")

        self.build()


    def build(self):
        self.batch_size = tf.placeholder_with_default(4, (), name="batch_size")

        self.x = tf.placeholder_with_default(
            tf.stop_gradient(self.base_distribution.sample(self.batch_size)),
            (None, 2),
            name="x")
        self.y = tf.placeholder_with_default(
            tf.stop_gradient(self.distribution.bijector.forward(self.x)),
            (None, 2),
            name="y")
        self.inverse_x = self.distribution.bijector.inverse(self.y)

        self.log_pi = self.distribution.log_prob(self.y)
        self.pi = tf.exp(self.log_pi)
        self._action = tf.tanh(self.pi) if self.config["squash"] else self.pi

        log_Z = 0.0
        self.Q = self._qf(self.y)
        surrogate_loss = tf.reduce_mean(
            self.log_pi * tf.stop_gradient(self.log_pi - self.Q + log_Z))

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)

        with tf.control_dependencies([tf.assert_equal(reg_loss, tf.constant(0.0))]):
            self.loss = surrogate_loss + reg_loss

        optimizer = tf.train.AdamOptimizer(
            self.config["learning_rate"], use_locking=False)

        self.train_op = optimizer.minimize(loss=self.loss)

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
