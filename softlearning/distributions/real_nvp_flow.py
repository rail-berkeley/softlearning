"""RealNVP bijector flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import bijectors
from tensorflow_probability.python.internal import tensorshape_util

from softlearning.models.feedforward import feedforward_model


__all__ = [
    "RealNVPFlow",
]


class RealNVPFlow(bijectors.Bijector):
    """A flow of RealNVP bijectors."""

    def __init__(self,
                 num_coupling_layers=2,
                 hidden_layer_sizes=(64,),
                 use_batch_normalization=False,
                 is_constant_jacobian=False,
                 validate_args=False,
                 name=None):
        """Instantiates the `RealNVPFlow` normalizing flow.

        Args:
            is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
                implementation assumes `log_scale` does not depend on the forward domain
                (`x`) or inverse domain (`y`) values. (No validation is made;
                `is_constant_jacobian=False` is always safe but possibly computationally
                inefficient.)
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str`, name given to ops managed by this object.
        """
        name = name or 'real_nvp_flow'

        self._num_coupling_layers = num_coupling_layers
        self._hidden_layer_sizes = tuple(hidden_layer_sizes)
        if use_batch_normalization:
            raise NotImplementedError(
                "TODO(hartikainen): Batch normalization is not yet supported"
                " for RealNVPFlow.")
        self._use_batch_normalization = use_batch_normalization

        self._built = False

        super(RealNVPFlow, self).__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=is_constant_jacobian,
            validate_args=validate_args,
            name=name)

    def _build(self, input_shape):
        input_depth = tf.compat.dimension_value(
            tensorshape_util.with_rank_at_least(input_shape, 1)[-1])

        self._input_depth = input_depth

        flow_parts = []
        for i in range(self._num_coupling_layers):
            if self._use_batch_normalization:
                batch_normalization_bijector = bijectors.BatchNormalization()
                flow_parts += [batch_normalization_bijector]

            real_nvp_bijector = bijectors.RealNVP(
                num_masked=input_depth // 2,
                shift_and_log_scale_fn=feedforward_scale_and_log_diag_fn(
                    hidden_layer_sizes=self._hidden_layer_sizes,
                    activation=tf.nn.relu),
                name='real_nvp_{}'.format(i))
            flow_parts += [real_nvp_bijector]

            if i < self._num_coupling_layers - 1:
                permute_bijector = bijectors.Permute(
                    permutation=list(reversed(range(input_depth))),
                    name='permute_{}'.format(i))
                flow_parts += [permute_bijector]

        # bijectors.Chain applies the list of bijectors in the
        # _reverse_ order of what they are inputted, thus [::-1].
        self.flow = bijectors.Chain(flow_parts[::-1])
        self._built = True

    def _forward(self, x, **condition_kwargs):
        if not self._built:
            self._build(x.shape)

        x = self.flow.forward(x, **condition_kwargs)
        return x

    def _inverse(self, y, **condition_kwargs):
        if not self._built:
            self._build(y.shape)

        y = self.flow.inverse(y, **condition_kwargs)
        return y

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        if not self._built:
            self._build(x.shape)

        fldj = self.flow.forward_log_det_jacobian(
            x, event_ndims=1, **condition_kwargs)
        return fldj

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        if not self._built:
            self._build(y.shape)

        ildj = self.flow.inverse_log_det_jacobian(
            y, event_ndims=1, **condition_kwargs)
        return ildj


def feedforward_scale_and_log_diag_fn(
        hidden_layer_sizes,
        shift_only=False,
        activation=tf.nn.relu,
        output_activation="linear",
        name=None,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):

    def _fn(x, output_units, **condition_kwargs):
        """MLP which concatenates the condition kwargs to input."""

        shift_and_log_scale = feedforward_model(
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=(1 if shift_only else 2) * output_units,
            activation=activation,
            output_activation=output_activation,
            name=name,
        )([x, condition_kwargs])

        if shift_only:
            return shift_and_log_scale, None

        shift, log_scale = tf.keras.layers.Lambda(
            lambda shift_and_scale: tf.split(shift_and_scale, 2, axis=-1)
        )(shift_and_log_scale)

        return shift, log_scale

    return _fn
