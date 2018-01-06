"""Real NVP policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.distributions import RealNVPBijector
from sac.policies import NNPolicy

EPS = 1e-6

DEFAULT_CONFIG = {
    "mode": "train",
    "D_in": 2,
    "learning_rate": 1e-4,
    "squash": False,
    "real_nvp_config": None
}

DEFAULT_BATCH_SIZE = 32

class RealNVPPolicy(NNPolicy, Serializable):
    """Real NVP policy"""

    def __init__(self,
                 env_spec,
                 config=None,
                 qf=None,
                 name="policy"):
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
        Serializable.quick_init(self, locals())

        self.config = dict(DEFAULT_CONFIG, **(config or {}))

        self._env_spec = env_spec
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._qf = qf

        self.name = name
        self.build()

        squash = self.config["squash"]
        super().__init__(
            env_spec,
            self._observations_ph,
            tf.tanh(self._actions) if squash else self._actions,
            'policy'
        )

    def actions_for(self, observations, name=None, reuse=tf.AUTO_REUSE):
        name = name or self.name
        with tf.variable_scope(name, reuse=reuse):
            N = tf.shape(observations)[0]
            return tf.stop_gradient(
                self.distribution.sample(
                    N, bijector_kwargs={"observations": observations}))

    def log_pi_for(self, observations, actions=None, name=None, reuse=tf.AUTO_REUSE):
        name = name or self.name
        if actions is None:
            actions = self.actions_for(observations, name, reuse)

        with tf.variable_scope(name, reuse=reuse):
            return self.distribution.log_prob(
                actions, bijector_kwargs={"observations": observations})

    def build(self):
        ds = tf.contrib.distributions
        self.bijector = RealNVPBijector(
            config=self.config["real_nvp_config"], event_ndims=self._Da)

        self.base_distribution = ds.MultivariateNormalDiag(
            loc=tf.zeros(self._Da), scale_diag=tf.ones(self._Da))

        self.distribution = ds.ConditionalTransformedDistribution(
            distribution=self.base_distribution,
            bijector=self.bijector,
            name="RealNVPPolicyDistribution")

        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        self._actions = self.actions_for(self._observations_ph)

    def get_action(self, observations):
        """Sample single action based on the observations.

        TODO: if self._is_deterministic
        """
        return self.get_actions(observations[None])[0], {}

    def get_actions(self, observations):
        """Sample batch of actions based on the observations"""

        feed_dict = {self._observations_ph: observations}
        actions = tf.get_default_session().run(
            self._actions, feed_dict=feed_dict)
        return actions

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        current = getattr(self, "_is_deterministic", None)
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = current

    def get_params_internal(self, **tags):
        if tags: raise NotImplementedError
        return tf.trainable_variables(scope=self.name)

    def reset(self, dones=None):
        pass

    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        TODO: implement
        """
        pass
