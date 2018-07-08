"""Latent Space Policy."""

from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd

from serializable import Serializable
from rllab.misc import logger

from softlearning.distributions import ConditionalRealNVPFlow
from softlearning.policies import NNPolicy

EPS = 1e-6


class LatentSpacePolicy(NNPolicy, Serializable):
    """Latent Space Policy."""

    def __init__(self,
                 observation_shape,
                 action_shape,
                 mode="train",
                 squash=True,
                 bijector_config=None,
                 reparameterize=False,
                 observations_preprocessor=None,
                 fix_h_on_reset=False,
                 Q=None,
                 n_map_action_candidates=100,
                 name="lsp_policy"):
        """Initialize LatentSpacePolicy.

        Args:
            observation_shape (`list`, `tuple`): Dimension of the observations.
            action_shape (`list`, `tuple`): Dimension of the actions.
            bijector_config (`dict`): Parameter configuration for bijector.
            squash (`bool`): If True, squash the action samples between
                -1 and 1 with tanh.
            n_map_action_candidates ('int'): Number of action candidates for
            estimating the maximum a posteriori (deterministic) action.
        """
        self._Serializable__initialize(locals())

        assert len(observation_shape) == 1, observation_shape
        self._Ds = observation_shape[0]
        assert len(action_shape) == 1, action_shape
        self._Da = action_shape[0]
        self._bijector_config = bijector_config
        self._mode = mode
        self._squash = squash
        self._reparameterize = reparameterize
        self._fix_h_on_reset = fix_h_on_reset
        self._Q = Q
        self._n_map_action_candidates = n_map_action_candidates

        self._fixed_h = None
        self._is_deterministic = False
        self._observations_preprocessor = observations_preprocessor

        self.NO_OP = tf.no_op()

        self.name = name
        self.build()

        super(NNPolicy, self).__init__(env_spec=None)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE, with_log_pis=False,
                    with_raw_actions=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            conditions = (
                self._observations_preprocessor(observations)
                if self._observations_preprocessor is not None
                else observations)

            if latents is None:
                shape = tf.shape(conditions)[0]
                latents = self.base_distribution.sample(shape)

            raw_actions = self.bijector.forward(
                latents, condition=conditions)

            if not self._reparameterize:
                raw_actions = tf.stop_gradient(raw_actions)

        actions = tf.tanh(raw_actions) if self._squash else raw_actions
        return_list = [actions]
        if with_log_pis:
            log_pis = self._log_pis_for_raw(conditions, raw_actions,
                                            name, reuse=reuse)
            return_list.append(log_pis)

        if with_raw_actions:
            return_list.append(raw_actions)

        # not sure the best way of returning variable outputs
        if len(return_list) > 1:
            return return_list

        return actions

    def _log_pis_for_raw(self, conditions, raw_actions, name=None,
                         reuse=tf.AUTO_REUSE):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            log_pis = self.distribution.log_prob(
                raw_actions, bijector_kwargs={"condition": conditions})

        if self._squash:
            log_pis -= self._squash_correction(raw_actions)

        return log_pis

    def log_pis_for(self, observations, raw_actions=None, actions=None, name=None,
                    reuse=tf.AUTO_REUSE):
        name = name or self.name

        assert raw_actions is not None or actions is not None

        with tf.variable_scope(name, reuse=reuse):
            conditions = (
                self._observations_preprocessor(observations)
                if self._observations_preprocessor is not None
                else observations)

        if raw_actions is not None:
            return self._log_pis_for_raw(conditions, raw_actions, name=name, reuse=reuse)

        if self._squash:
            actions = tf.atanh(actions)
        return self._log_pis_for_raw(conditions, actions, name=name, reuse=reuse)

    def build(self):
        config = self._bijector_config
        self.bijector = ConditionalRealNVPFlow(
            num_coupling_layers=config.get("num_coupling_layers"),
            translation_hidden_sizes=config.get("translation_hidden_sizes"),
            scale_hidden_sizes=config.get("scale_hidden_sizes"),
            event_ndims=1,
            event_dims=(self._Da, ))

        self.base_distribution = tfpd.MultivariateNormalDiag(
            loc=tf.zeros(self._Da), scale_diag=tf.ones(self._Da))
        # TODO(hartikainen): Without setting _is_constant_jacobian, the
        # tfpd.MultivariateNormalDiag.sample(N) caches the samples based on
        # `event_dims` and not `N`. Because we call sample with
        # N = tf.shape(observation)[0], things break if we use two different
        # observations, e.g. one in SAC and one in LatentSpacePolicy.
        # TODO(hartikainen): Probably should file an issue to TensorFlow
        # probability about this.
        self.base_distribution.bijector._is_constant_jacobian = False

        self.sample_z = self.base_distribution.sample(1)

        self.distribution = (
            tfpd.ConditionalTransformedDistribution(
                distribution=self.base_distribution,
                bijector=self.bijector,
                name="lsp_distribution"))

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

        self._actions, self._log_pis, self._raw_actions = self.actions_for(
            self._observations_ph, with_log_pis=True, with_raw_actions=True)
        self._det_actions, self._det_actions_raw = self.actions_for(
            self._observations_ph, self._latents_ph, with_raw_actions=True)

    def get_action(self,
                   observation,
                   with_log_pis=False,
                   with_raw_actions=False):
        """Sample single action based on the observations."""

        if self._is_deterministic and self._n_map_action_candidates > 1:
            observations = np.tile(
                observation[None], reps=(self._n_map_action_candidates, 1))

            assert not with_log_pis, "No log_pis for deterministic action"

            (action_candidates,
             log_pis,
             raw_action_candidates) = self.get_actions(
                observations,
                with_log_pis=False,
                with_raw_actions=with_raw_actions)
            Q_values = self._Q.predict((observations, action_candidates))

            best_action_index = np.argmax(Q_values)
            best_action = action_candidates[best_action_index]
            best_raw_action = (
                raw_action_candidates[best_action_index]
                if with_raw_actions
                else None)

            return (best_action, log_pis, best_raw_action), {}

        return super(LatentSpacePolicy, self).get_action(
            observation, with_log_pis, with_raw_actions)

    def get_actions(self,
                    observations,
                    with_log_pis=False,
                    with_raw_actions=False):
        """Sample batch of actions based on the observations"""

        if self._fixed_h is not None:
            feed_dict = {self._observations_ph: observations}
            latents = np.tile(self._fixed_h,
                              reps=(self._n_map_action_candidates, 1))
            feed_dict.update({self._latents_ph: latents})

            assert not with_log_pis, "No log_pis for deterministic action"

            fetches = (
                self._det_actions,
                self.NO_OP,
                self._det_actions_raw if with_raw_actions else self.NO_OP)

            return tf.get_default_session().run(fetches, feed_dict=feed_dict)

        return super(LatentSpacePolicy, self).get_actions(
            observations, with_log_pis, with_raw_actions)

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
            if h is None:
                h = self.sample_z.eval()
            self._fixed_h = h

        yield

        self._is_deterministic = was_deterministic
        self._fixed_h = old_fixed_h

    def reset(self, dones=None):
        if self._fix_h_on_reset:
            self._fixed_h = self.sample_z.eval()

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger."""

        feeds = {self._observations_ph: batch['observations']}
        raw_actions, actions, log_pis = tf.get_default_session().run(
            (self._raw_actions, self._actions, self._log_pis), feeds)

        logger.record_tabular('-log-pi-mean', np.mean(-log_pis))
        logger.record_tabular('-log-pi-min', np.min(-log_pis))
        logger.record_tabular('-log-pi-max', np.max(-log_pis))
        logger.record_tabular('-log-pi-std', np.std(-log_pis))

        logger.record_tabular('actions-mean', np.mean(actions))
        logger.record_tabular('actions-min', np.min(actions))
        logger.record_tabular('actions-max', np.max(actions))
        logger.record_tabular('actions-std', np.std(actions))

        logger.record_tabular('raw-actions-mean', np.mean(raw_actions))
        logger.record_tabular('raw-actions-min', np.min(raw_actions))
        logger.record_tabular('raw-actions-max', np.max(raw_actions))
        logger.record_tabular('raw-actions-std', np.std(raw_actions))
