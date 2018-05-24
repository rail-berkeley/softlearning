"""Latent Space Policy."""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger

from sac.distributions import RealNVPBijector
from sac.policies import NNPolicy


EPS = 1e-6


class LatentSpacePolicy(NNPolicy, Serializable):
    """Latent Space Policy."""

    def __init__(self,
                 env_spec,
                 mode="train",
                 squash=True,
                 bijector_config=None,
                 observations_preprocessor=None,
                 fix_h_on_reset=False,
                 q_function=None,
                 n_map_action_candidates=100,
                 name="lsp_policy"):
        """Initialize LatentSpacePolicy.

        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            bijector_config (`dict`): Parameter configuration for bijector.
            squash (`bool`): If True, squash the action samples between
                -1 and 1 with tanh.
            n_map_action_candidates ('int'): Number of action candidates for
            estimating the maximum a posteriori (deterministic) action.
        """
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._bijector_config = bijector_config
        self._mode = mode
        self._squash = squash
        self._fix_h_on_reset = fix_h_on_reset
        self._q_function = q_function
        self._n_map_action_candidates=n_map_action_candidates

        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._fixed_h = None
        self._is_deterministic = False
        self._observations_preprocessor = observations_preprocessor

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")
        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE, with_log_pis=False,
                    with_raw_actions=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            conditions = (
                self._observations_preprocessor.get_output_for(
                    observations, reuse=reuse)
                if self._observations_preprocessor is not None
                else observations)

            if latents is not None:
                raw_actions = self.bijector.forward(
                    latents, condition=conditions)
            else:
                N = tf.shape(conditions)[0]
                raw_actions = self.distribution.sample(
                    N, bijector_kwargs={"condition": conditions})

            raw_actions = tf.stop_gradient(raw_actions)

        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            log_pis = self.log_pis_for(
                conditions, raw_actions, name=name, reuse=reuse)

            if with_raw_actions:
                return raw_actions, actions, log_pis

            return actions, log_pis

        return actions

    def log_pis_for(self, conditions, raw_actions, name=None,
                    reuse=tf.AUTO_REUSE):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            log_pis = self.distribution.log_prob(
                raw_actions, bijector_kwargs={"condition": conditions})

        if self._squash:
            log_pis -= self._squash_correction(raw_actions)

        return log_pis

    def build(self):
        ds = tf.contrib.distributions
        config = self._bijector_config
        self.bijector = RealNVPBijector(
            num_coupling_layers=config.get("num_coupling_layers"),
            translation_hidden_sizes=config.get("translation_hidden_sizes"),
            scale_hidden_sizes=config.get("scale_hidden_sizes"),
            event_ndims=self._Da)

        self.base_distribution = ds.MultivariateNormalDiag(
            loc=tf.zeros(self._Da), scale_diag=tf.ones(self._Da))

        self.sample_z = self.base_distribution.sample(1)

        self.distribution = ds.ConditionalTransformedDistribution(
            distribution=self.base_distribution,
            bijector=self.bijector,
            name="lsp_distribution")

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

        self._raw_actions, self._actions, self._log_pis = self.actions_for(
            self._observations_ph, with_log_pis=True, with_raw_actions=True)
        self._determistic_actions = self.actions_for(self._observations_ph,
                                                     self._latents_ph)

    def get_action(self, observation):
        """Sample single action based on the observations.
        """

        if self._is_deterministic and self._n_map_action_candidates > 1:
            observations = np.tile(
                observation[None], reps=(self._n_map_action_candidates, 1))
            action_candidates = self.get_actions(observations)
            q_values = self._q_function.eval(observations, action_candidates)
            best_action_index = np.argmax(q_values)

            return action_candidates[best_action_index], {}
        return self.get_actions(observation[None])[0], {}

    def get_actions(self, observations):
        """Sample batch of actions based on the observations"""

        feed_dict = { self._observations_ph: observations }

        if self._fixed_h is not None:
            latents = np.tile(self._fixed_h,
                              reps=(self._n_map_action_candidates, 1))
            feed_dict.update({ self._latents_ph: latents })
            actions = tf.get_default_session().run(
                self._determistic_actions,
                feed_dict=feed_dict)
        else:
            actions = tf.get_default_session().run(
                self._actions, feed_dict=feed_dict)

        return actions

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True, h=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
            to during the context. The value will be reset back to the previous
            value when the context exits.
        """
        was_deterministic = self._is_deterministic
        old_fixed_h = self._fixed_h

        self._is_deterministic = set_deterministic
        if set_deterministic:
            if h is None: h = self.sample_z.eval()
            self._fixed_h = h

        yield

        self._is_deterministic = was_deterministic
        self._fixed_h = old_fixed_h

    def get_params_internal(self, **tags):
        if tags: raise NotImplementedError
        return tf.trainable_variables(scope=self._scope_name)

    def reset(self, dones=None):
        if self._fix_h_on_reset:
            self._fixed_h = self.sample_z.eval()

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger."""

        feeds = { self._observations_ph: batch['observations'] }
        raw_actions, actions, log_pis = tf.get_default_session().run(
            (self._raw_actions, self._actions, self._log_pis), feeds)

        logger.record_tabular('policy-entropy-mean', -np.mean(log_pis))
        logger.record_tabular('log-pi-min', np.min(log_pis))
        logger.record_tabular('log-pi-max', np.max(log_pis))

        logger.record_tabular('actions-mean', np.mean(actions))
        logger.record_tabular('actions-min', np.min(actions))
        logger.record_tabular('actions-max', np.max(actions))

        logger.record_tabular('raw-actions-mean', np.mean(raw_actions))
        logger.record_tabular('raw-actions-min', np.min(raw_actions))
        logger.record_tabular('raw-actions-max', np.max(raw_actions))
