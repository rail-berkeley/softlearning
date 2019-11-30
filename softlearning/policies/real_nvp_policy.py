"""RealNVPPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.distributions.real_nvp_flow import RealNVPFlow
from softlearning.utils.tensorflow import nest

from .base_policy import LatentSpacePolicy


class RealNVPPolicy(LatentSpacePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 action_range,
                 *args,
                 squash=True,
                 preprocessors=None,
                 hidden_layer_sizes=(128, 128),
                 num_coupling_layers=2,
                 name=None,
                 **kwargs):

        raise NotImplementedError(
            "TODO(hartikainen): RealNVPPolicy is currently broken. The keras"
            " models together with the tfp distributions somehow count the"
            "variables multiple times. This needs to be fixed before usage.")
        assert (np.all(action_range == np.array([[-1], [1]]))), (
            "The action space should be scaled to (-1, 1)."
            " TODO(hartikainen): We should support non-scaled actions spaces.")

        self._Serializable__initialize(locals())

        self._action_range = action_range
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name

        super(RealNVPPolicy, self).__init__(*args, **kwargs)

        inputs_flat = create_inputs(input_shapes)
        preprocessors_flat = (
            flatten_input_structure(preprocessors)
            if preprocessors is not None
            else tuple(None for _ in inputs_flat))

        assert len(inputs_flat) == len(preprocessors_flat), (
            inputs_flat, preprocessors_flat)

        preprocessed_inputs = [
            preprocessor(input_) if preprocessor is not None else input_
            for preprocessor, input_
            in zip(preprocessors_flat, inputs_flat)
        ]

        def cast_and_concat(x):
            x = nest.map_structure(
                lambda element: tf.cast(element, tf.float32), x)
            x = nest.flatten(x)
            x = tf.concat(x, axis=-1)
            return x

        conditions = tf.keras.layers.Lambda(
            cast_and_concat
        )(preprocessed_inputs)

        self.condition_inputs = inputs_flat

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(input=x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(output_shape),
            scale_diag=tf.ones(output_shape))

        flow_model = RealNVPFlow(
            num_coupling_layers=num_coupling_layers,
            hidden_layer_sizes=hidden_layer_sizes)

        flow_distribution = flow_model(base_distribution)

        latents = base_distribution.sample(batch_size)

        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(
            shape=output_shape, name='latents')

        raw_actions = flow_distribution.bijector.forward(
            latents, conditions=conditions)

        raw_actions_for_fixed_latents = flow_distribution.bijector.forward(
            self.latents_input, conditions=conditions)

        squash_bijector = (
            tfp.bijectors.Tanh()
            if self._squash
            else tfp.bijectors.Identity())

        actions = squash_bijector(raw_actions)
        self.actions_model = tf.keras.Model(self.condition_inputs, actions)

        actions_for_fixed_latents = squash_bijector(raw_actions)
        self.actions_model_for_fixed_latents = tf.keras.Model(
            (*self.condition_inputs, self.latents_input),
            actions_for_fixed_latents)

        self.deterministic_actions_model = self.actions_model

        self.actions_input = tf.keras.layers.Input(
            shape=output_shape, name='actions')

        log_pis = flow_distribution.log_prob(actions)[..., tf.newaxis]
        log_pis_for_action_input = flow_distribution.log_prob(
            self.actions_input)[..., tf.newaxis]

        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input),
            log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (log_pis, raw_actions, actions))

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model.predict(inputs)

        return OrderedDict((
            ('-log-pis-mean', np.mean(-log_pis_np)),
            ('-log-pis-std', np.std(-log_pis_np)),

            ('raw-actions-mean', np.mean(raw_actions_np)),
            ('raw-actions-std', np.std(raw_actions_np)),

            ('actions-mean', np.mean(actions_np)),
            ('actions-std', np.std(actions_np)),
            ('actions-min', np.min(actions_np)),
            ('actions-max', np.max(actions_np)),
        ))


class FeedforwardRealNVPPolicy(RealNVPPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args,
                 **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardRealNVPPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net
