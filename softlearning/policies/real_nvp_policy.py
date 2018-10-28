from collections import OrderedDict
from contextlib import contextmanager


import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.distributions.real_nvp_flow import ConditionalRealNVPFlow
from .gaussian_policy import SquashBijector


class RealNVPPolicy(object):
    """TODO(hartikainen): Implement regularization"""

    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 bijector_config=None,
                 name=None,
                 *args,
                 **kwargs):
        self._squash = squash
        self._bijector_config = bijector_config
        self._is_deterministic = False

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = (
            tf.keras.layers.Concatenate(axis=-1)(self.condition_inputs)
            if len(self.condition_inputs) > 1
            else self.condition_inputs[0])

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0])(conditions)

        squash_bijector = (
            SquashBijector()
            if self._squash
            else tfp.bijectors.Identity())

        def actions_and_log_pis_fn(inputs):
            batch_size = inputs

            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(output_shape),
                scale_diag=tf.ones(output_shape))

            real_nvp_flow = ConditionalRealNVPFlow(
                **self._bijector_config,
                event_dims=output_shape)

            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=real_nvp_flow))

            raw_actions = distribution.sample(batch_size)
            log_pis = distribution.log_prob(raw_actions)
            actions = squash_bijector.forward(raw_actions)

            return [actions, raw_actions, log_pis[:, None]]

        actions, raw_actions, log_pis = tf.keras.layers.Lambda(
            actions_and_log_pis_fn)(batch_size)

        self.actions_and_log_pis_model = tf.keras.Model(
            conditions, [actions, raw_actions, log_pis])

    @property
    def trainable_variables(self):
        return self.actions_and_log_pis_model.trainable_variables

    @contextmanager
    def deterministic(self, set_deterministic=True):
        was_deterministic = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = was_deterministic

    def reset(self):
        pass

    def actions_and_log_pis(self, conditions):
        if self._is_deterministic:
            return self.deterministic_actions_and_log_pis_model(conditions)
        else:
            return self.actions_and_log_pis_model(conditions)

    def actions_and_log_pis_np(self, conditions):
        if self._is_deterministic:
            return self.deterministic_actions_and_log_pis_model.predict(
                conditions)
        else:
            return self.actions_and_log_pis_model.predict(conditions)

    def get_diagnostics(self, iteration, batch):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        return OrderedDict({})
