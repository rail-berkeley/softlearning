"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from softlearning.distributions.squash_bijector import SquashBijector
from softlearning.models.feedforward import feedforward_model

from .base_policy import LatentSpacePolicy


SCALE_DIAG_MIN_MAX = (-20, 2)


class GaussianPolicy(LatentSpacePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 preprocessor=None,
                 name=None,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        shift_and_log_scale_diag = self._shift_and_log_scale_diag_net(
            input_shapes=(conditions.shape[1:], ),
            output_size=output_shape[0] * 2,
        )(conditions)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda shift_and_log_scale_diag: tf.split(
                shift_and_log_scale_diag,
                num_or_size_splits=2,
                axis=-1)
        )(shift_and_log_scale_diag)

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

        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(shape=output_shape)

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

        raw_actions_for_fixed_latents = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, self.latents_input))

        squash_bijector = (
            SquashBijector()
            if self._squash
            else tfp.bijectors.Identity())

        actions = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        actions_for_fixed_latents = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions_for_fixed_latents)
        self.actions_model_for_fixed_latents = tf.keras.Model(
            (*self.condition_inputs, self.latents_input),
            actions_for_fixed_latents)

        deterministic_actions = tf.keras.layers.Lambda(
            lambda shift: squash_bijector.forward(shift)
        )(shift)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)

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
            log_pis_fn)([shift, log_scale_diag, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, self.actions_input])

        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input),
            log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (shift, log_scale_diag, log_pis, raw_actions, actions))

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(GaussianPolicy, self).non_trainable_weights))

    def actions(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model(conditions)

        return self.actions_model(conditions)

    def log_pis(self, conditions, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model([*conditions, actions])

    def actions_np(self, conditions):
        return super(GaussianPolicy, self).actions_np(conditions)

    def log_pis_np(self, conditions, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model.predict([*conditions, actions])

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np,
         log_scale_diags_np,
         log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict((
            ('shifts-mean', np.mean(shifts_np)),
            ('shifts-std', np.std(shifts_np)),

            ('log_scale_diags-mean', np.mean(log_scale_diags_np)),
            ('log_scale_diags-std', np.std(log_scale_diags_np)),

            ('-log-pis-mean', np.mean(-log_pis_np)),
            ('-log-pis-std', np.std(-log_pis_np)),

            ('raw-actions-mean', np.mean(raw_actions_np)),
            ('raw-actions-std', np.std(raw_actions_np)),

            ('actions-mean', np.mean(actions_np)),
            ('actions-std', np.std(actions_np)),
            ('actions-min', np.min(actions_np)),
            ('actions-max', np.max(actions_np)),
        ))


class FeedforwardGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net
