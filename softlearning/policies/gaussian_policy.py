"""GaussianPolicy."""

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from serializable import Serializable

from softlearning.distributions import Normal
from softlearning.policies import NNPolicy


class SquashBijector(tfp.bijectors.Tanh):
    """Bijector similar to Tanh-bijector, but with stable log det jacobian."""

    def _forward_log_det_jacobian(self, x):
        return 2.0 * (tf.log(2.0) - x - tf.nn.softplus(-2.0 * x))


class GaussianPolicy(NNPolicy, Serializable):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 hidden_layer_sizes=(100, 100),
                 reg=1e-3,
                 squash=True,
                 reparameterize=True,
                 name='gaussian_policy'):
        """
        Args:
            observation_shape (`list`, `tuple`): Dimension of the observations.
            action_shape (`list`, `tuple`): Dimension of the actions.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian
                parameters.
            squash (`bool`): If True, squash the Gaussian action samples
                between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly
                through the action samples.
        """
        self._Serializable__initialize(locals())

        self._hidden_layers = hidden_layer_sizes
        assert len(observation_shape) == 1, observation_shape
        self._Ds = observation_shape[0]
        assert len(action_shape) == 1, action_shape
        self._Da = action_shape[0]
        self._is_deterministic = False
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg

        self.name = name
        self.NO_OP = tf.no_op()

        self.build()

        super(NNPolicy, self).__init__(env_spec=None)

    def actions_for(self, observations, name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, with_raw_actions=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        raw_actions = distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions
        return_list = [actions]

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            log_pis = self._log_pis_for_raw(observations, raw_actions,
                                            name)
            return_list.append(log_pis)

        if with_raw_actions:
            return_list.append(raw_actions)

        # not sure the best way of returning variable outputs
        if len(return_list) > 1:
            return return_list

        return actions

    def _log_pis_for_raw(self, observations, actions, name=None,
                         reuse=tf.AUTO_REUSE):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        log_pis = distribution.log_prob(actions)
        if self._squash:
            log_pis -= self._squash_correction(actions)
        return log_pis

    def log_pis_for(self, observations, raw_actions=None, actions=None, name=None,
                    reuse=tf.AUTO_REUSE):
        assert raw_actions is not None or actions is not None, 'Must provide either actions or raw_actions'

        # we prefer to use raw actions as to avoid instability with atanh
        if raw_actions is not None:
            return self._log_pis_for_raw(observations, raw_actions, name, reuse)
        if self._squash:
            actions = tf.atanh(actions)
        return self._log_pis_for_raw(observations, actions, name,
                                     reuse)

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )
        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observations_ph,),
                reg=self._reg,
            )

        self._actions, self._log_pis, self._raw_actions = self.actions_for(
            self._observations_ph, with_log_pis=True, with_raw_actions=True)

    def get_actions(self, observations, with_log_pis=False, with_raw_actions=False):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the
        observations. If False, return stochastically sampled action.

        """
        if self._is_deterministic:  # Handle the deterministic case separately.

            feed_dict = {self._observations_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            raw_mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict)  # 1 x Da
            mu = np.tanh(raw_mu) if self._squash else raw_mu

            assert not with_log_pis, "No log_pis for deterministic action"

            if with_raw_actions:
                return mu, None, raw_mu

            return mu, None, None

        return super(GaussianPolicy, self).get_actions(
            observations, with_log_pis, with_raw_actions)

    @contextmanager
    def deterministic(self, set_deterministic=True):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    def get_diagnostics(self, iteration, batch):
        """Record diagnostic information.

        Records the mean, min, max, and standard deviation of means and
        covariances.
        """

        feeds = {self._observations_ph: batch['observations']}
        sess = tf.keras.backend.get_session()
        actions, raw_actions, log_pi, mu, log_sig, = sess.run(
            (
                self._actions,
                self._raw_actions,
                self._log_pis,
                self.distribution.mu_t,
                self.distribution.log_sig_t,
            ),
            feeds
        )

        diagnostics = OrderedDict({
            'policy-mus-mean': np.mean(mu),
            'policy-mus-min': np.min(mu),
            'policy-mus-max': np.max(mu),
            'policy-mus-std': np.std(mu),

            'log-sigs-mean': np.mean(log_sig),
            'log-sigs-min': np.min(log_sig),
            'log-sigs-max': np.max(log_sig),
            'log-sigs-std': np.std(log_sig),

            '-log-pi-mean': np.mean(-log_pi),
            '-log-pi-max': np.max(-log_pi),
            '-log-pi-min': np.min(-log_pi),
            '-log-pi-std': np.std(-log_pi),

            'actions-mean': np.mean(actions),
            'actions-min': np.min(actions),
            'actions-max': np.max(actions),
            'actions-std': np.std(actions),

            'raw-actions-mean': np.mean(raw_actions),
            'raw-actions-min': np.min(raw_actions),
            'raw-actions-max': np.max(raw_actions),
            'raw-actions-std': np.std(raw_actions),
        })

        return diagnostics


SCALE_DIAG_MIN_MAX = (-20, 2)


class GaussianPolicyV2(object):
    """TODO(hartikainen): Implement regularization"""

    def __init__(self,
                 input_shapes,
                 output_shape,
                 hidden_layer_sizes,
                 squash=True,
                 regularization_coeff=1e-3,
                 activation='relu',
                 output_activation='linear',
                 name=None,
                 *args,
                 **kwargs):
        self._squash = squash
        self._regularization_coeff = regularization_coeff

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = (
            tf.keras.layers.Concatenate(axis=-1)(self.condition_inputs)
            if len(self.condition_inputs) > 1
            else self.condition_inputs[0])

        out = conditions
        for units in hidden_layer_sizes:
            out = tf.keras.layers.Dense(
                units, *args, activation=activation, **kwargs)(out)

        out = tf.keras.layers.Dense(
            output_shape[0] * 2, *args,
            activation=output_activation, **kwargs
        )(out)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda shift_and_log_scale_diag: tf.split(
                shift_and_log_scale_diag,
                num_or_size_splits=2,
                axis=-1)
        )(out)

        log_scale_diag = tf.keras.layers.Lambda(
            lambda log_scale_diag: tf.clip_by_value(
                log_scale_diag, *SCALE_DIAG_MIN_MAX)
        )(log_scale_diag)

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(output_shape),
            scale_diag=tf.ones(output_shape))

        latents = tf.keras.layers.Lambda(
            lambda batch_size: base_distribution.sample(batch_size)
        )(batch_size)

        def raw_actions_fn(inputs):
            shift, log_scale_diag, latents = inputs
            bijector = tfp.bijectors.Affine(
                shift=shift,
                scale_diag=tf.exp(log_scale_diag))
            actions = bijector.forward(latents)
            return actions

        raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, latents))

        squash_bijector = (
            SquashBijector()
            if self._squash
            else tfp.bijectors.Identity())

        actions = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions)

        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        def log_pis_fn(inputs):
            shift, log_scale_diag, actions = inputs
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(output_shape),
                scale_diag=tf.ones(output_shape))
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=shift,
                    scale_diag=tf.exp(log_scale_diag)),
            ))
            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector))

            log_pis = distribution.log_prob(actions)[:, None]
            return log_pis

        self.actions_input = tf.keras.layers.Input(shape=output_shape)

        log_pis = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, self.actions_input])

        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input), log_pis)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    def reset(self):
        pass

    def actions(self, conditions):
        return self.actions_model(conditions)

    def log_pis(self, conditions, actions):
        return self.log_pis_model([*conditions, actions])

    def actions_np(self, conditions):
        return self.actions_model.predict(conditions)

    def log_pis_np(self, conditions, actions):
        return self.log_pis_model.predict([*conditions, actions])

    def get_diagnostics(self, iteration, batch):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        return OrderedDict({})
