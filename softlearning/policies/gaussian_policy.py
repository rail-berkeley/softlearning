"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.distributions.bijectors import (
    ConditionalShift, ConditionalScale)

from .base_policy import LatentSpacePolicy


class GaussianPolicy(LatentSpacePolicy):
    def __init__(self, *args, **kwargs):
        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.shift_and_scale_model = self._shift_and_log_scale_diag_net(
            inputs=self.inputs,
            output_size=np.prod(self._output_shape) * 2)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self._output_shape),
            scale_diag=tf.ones(self._output_shape))

        raw_action_distribution = tfp.bijectors.Chain((
            ConditionalShift(name='shift'),
            ConditionalScale(name='scale'),
        ))(base_distribution)

        self.base_distribution = base_distribution
        self.raw_action_distribution = raw_action_distribution
        self.action_distribution = self._action_post_processor(
            raw_action_distribution)

    # @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        """Compute actions for given observations.

        # TODO(hartikainen): This should have `@tf.function` decorator
        around it, but is currently disabled due to erroneous behavior with it.
        """
        observations = self._filter_observations(observations)

        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        shifts, scales = self.shift_and_scale_model(observations)
        actions = self.action_distribution.sample(
            batch_shape,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}})

        return actions

    # @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        """Compute log probabilities of `actions` given observations.

        # TODO(hartikainen): This should have `@tf.function` decorator
        around it, but is currently disabled due to erroneous behavior with it.
        """
        observations = self._filter_observations(observations)

        shifts, scales = self.shift_and_scale_model(observations)
        log_probs = self.action_distribution.log_prob(
            actions,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}}
        )[..., tf.newaxis]

        return log_probs

    # @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        """Compute probabilities of `actions` given observations.

        # TODO(hartikainen): This should have `@tf.function` decorator
        around it, but is currently disabled due to erroneous behavior with it.
        """
        observations = self._filter_observations(observations)
        shifts, scales = self.shift_and_scale_model(observations)
        probs = self.action_distribution.prob(
            actions,
            bijector_kwargs={'scale': {'scale': scales},
                             'shift': {'shift': shifts}}
        )[..., tf.newaxis]

        return probs

    def _shift_and_log_scale_diag_net(self, inputs, output_size):
        raise NotImplementedError

    def save_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.load_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.shift_and_scale_model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.shift_and_scale_model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.shift_and_scale_model.non_trainable_weights

    # @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        shifts, scales = self.shift_and_scale_model(inputs)
        actions = self.actions(inputs)
        log_pis = self.log_probs(inputs, actions)

        return OrderedDict((
            ('shifts-mean', tf.reduce_mean(shifts)),
            ('shifts-std', tf.math.reduce_std(shifts)),

            ('scales-mean', tf.reduce_mean(scales)),
            ('scales-std', tf.math.reduce_std(scales)),

            ('entropy-mean', tf.reduce_mean(-log_pis)),
            ('entropy-std', tf.math.reduce_std(-log_pis)),

            ('actions-mean', tf.reduce_mean(actions)),
            ('actions-std', tf.math.reduce_std(actions)),
            ('actions-min', tf.reduce_min(actions)),
            ('actions-max', tf.reduce_max(actions)),
        ))


class FeedforwardGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args,
                 **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, inputs, output_size):
        preprocessed_inputs = self._preprocess_inputs(inputs)
        shift_and_log_scale_diag = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_shape=[output_size],
            activation=self._activation,
            output_activation=self._output_activation
        )(preprocessed_inputs)

        def split_shift_and_log_scale_diag_fn(inputs):
            shift_and_log_scale_diag = inputs
            shift, log_scale_diag = tf.split(
                shift_and_log_scale_diag,
                num_or_size_splits=2,
                axis=-1)
            scale_diag = tf.exp(log_scale_diag)
            return [shift, scale_diag]

        shift, scale = tf.keras.layers.Lambda(
            split_shift_and_log_scale_diag_fn
        )(shift_and_log_scale_diag)
        shift_and_scale_diag_model = tf.keras.Model(
            inputs, (shift, scale))

        return shift_and_scale_diag_model

    def get_config(self):
        base_config = super(FeedforwardGaussianPolicy, self).get_config()
        config = {
            **base_config,
            'hidden_layer_sizes': self._hidden_layer_sizes,
            'activation': self._activation,
            'output_activation': self._output_activation,
        }
        return config
