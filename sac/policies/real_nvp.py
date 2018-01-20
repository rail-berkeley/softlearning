"""Real NVP policy"""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.distributions import RealNVPBijector
from sac.policies import NNPolicy

class RealNVPPolicy(NNPolicy, Serializable):
    """Real NVP policy"""

    def __init__(self,
                 env_spec,
                 mode="train",
                 squash=True,
                 real_nvp_config=None,
                 observations_preprocessor=None,
                 name="real_nvp_policy"):
        """Initialize Real NVP policy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            real_nvp_config (`dict`): Parameter
                configuration for real nvp distribution.
            squash (`bool`): If True, squash the action samples between
                -1 and 1 with tanh.
        """
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._real_nvp_config = real_nvp_config
        self._mode = mode
        self._squash = squash
        self._observations_preprocessor = observations_preprocessor

        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._fixed_h = None
        self._is_deterministic = False
        self._observations_preprocessor = observations_preprocessor

        self.name = name
        self.build()

        super().__init__(
            env_spec,
            self._observations_ph,
            tf.tanh(self._actions) if squash else self._actions,
            scope_name='policy'
        )

    def actions_for(self, observations, name=None, reuse=tf.AUTO_REUSE,
                    stop_gradient=True):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            if self._observations_preprocessor is not None:
                condition_var = self._observations_preprocessor.get_output_for(
                    observations, reuse=reuse)
            else:
                condition_var = observations

            N = tf.shape(condition_var)[0]
            actions = self.distribution.sample(
                N, bijector_kwargs={"observations": condition_var})

            if stop_gradient:
                actions = tf.stop_gradient(actions)

            return actions


    def log_pi_for(self, condition_var, actions=None, name=None, reuse=tf.AUTO_REUSE,
                   stop_action_gradient=True):
        name = name or self.name
        if actions is None:
            actions = self.actions_for(condition_var, name, reuse,
                                       stop_gradient=stop_action_gradient)

        with tf.variable_scope(name, reuse=reuse):
            if self._observations_preprocessor is not None:
                condition_var = self._observations_preprocessor.get_output_for(
                    condition_var, reuse=reuse)

            return self.distribution.log_prob(
                actions, bijector_kwargs={"observations": condition_var})

    def build(self):
        ds = tf.contrib.distributions
        real_nvp_config = self._real_nvp_config
        self.bijector = RealNVPBijector(
            num_coupling_layers=real_nvp_config.get("num_coupling_layers"),
            translation_hidden_sizes=real_nvp_config.get("translation_hidden_sizes"),
            scale_hidden_sizes=real_nvp_config.get("scale_hidden_sizes"),
            scale_regularization=real_nvp_config.get("scale_regularization"),
            event_ndims=self._Da)

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

        self._latents_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

        if self._observations_preprocessor is not None:
            self._condition_var = self._observations_preprocessor.get_output_for(
                self._observations_ph, reuse=True)
        else:
            self._condition_var = self._observations_ph

        self._actions = self.actions_for(self._observations_ph)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._determistic_actions = self.bijector.forward(
                self._latents_ph, observations=self._condition_var)

    def get_action(self, observation):
        """Sample single action based on the observations.

        TODO: if self._is_deterministic
        """
        return self.get_actions(observation[None])[0], {}

    def get_actions(self, observations):
        """Sample batch of actions based on the observations"""

        feed_dict = { self._observations_ph: observations }

        if self._fixed_h is not None:
            feed_dict.update({
                self._latents_ph: self._fixed_h
            })
            actions = tf.get_default_session().run(
                self._determistic_actions,
                feed_dict=feed_dict)
        else:
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
        was_deterministic = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = was_deterministic

    @contextmanager
    def fix_h(self, h=None):
        if h is None:
            h = self.base_distribution.sample(1).eval()

        print("h:", h)
        was_deterministic = self._is_deterministic
        self._is_deterministic = True
        self._fixed_h = h
        yield
        self._fixed_h = None
        self._is_deterministic = was_deterministic

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
