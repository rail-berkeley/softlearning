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


class FeedforwardBijectorFunction(tf.Module):
    def __init__(self,
                 hidden_layer_sizes,
                 shift_only=False,
                 activation=tf.nn.relu,
                 output_activation="linear",
                 name=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shift_only = shift_only
        self.activation = activation
        self.output_activation = output_activation
        self._built = False

    def __call__(self, x, output_units, **condition_kwargs):
        if not self._built:
            self._shift_and_log_scale_model = feedforward_model(
                hidden_layer_sizes=self.hidden_layer_sizes,
                output_shape=[(1 if self.shift_only else 2) * output_units],
                activation=self.activation,
                output_activation=self.output_activation)
            self._built = True

        # condition_kwargs is a dict, but feedforward_model implicitly flattens
        # these values. Effectively the same as
        # self._shift_and_log_scale_model(tree.flatten((x, condition_kwargs)))
        shift_and_log_scale = self._shift_and_log_scale_model(
            (x, condition_kwargs))

        # It would be nice to have these be encapsulated in the
        # `self._shift_and_log_scale_model`, but the issue is that
        # `tf.keras.Sequential` can't return tuples/lists, and functional
        # model type would have to know the input shape in advance.
        # The correct way here would be to create a subclassed model and
        # instantiate the model in the `build` method.
        shift, log_scale = tf.keras.layers.Lambda(
            lambda x: tf.split(x, 2, axis=-1)
        )(shift_and_log_scale)
        bijector = bijectors.affine_scalar.AffineScalar(
            shift=shift, log_scale=log_scale)
        return bijector


class RealNVPFlow(bijectors.Bijector):
    """A flow of RealNVP bijectors."""

    def __init__(self,
                 num_coupling_layers=2,
                 hidden_layer_sizes=(64,),
                 activation=tf.nn.relu,
                 use_batch_normalization=False,
                 is_constant_jacobian=False,
                 validate_args=False,
                 name='real_nvp_flow'):
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
        self._num_coupling_layers = num_coupling_layers
        self._hidden_layer_sizes = tuple(hidden_layer_sizes)
        self._activation = activation
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
                # TODO(hartikainen): Allow other normalizations, e.g.
                # weight normalization?
                batch_normalization_bijector = bijectors.BatchNormalization()
                flow_parts += [batch_normalization_bijector]

            real_nvp_bijector = bijectors.RealNVP(
                fraction_masked={True: 1.0, False: -1.0}[i % 2 == 0] * 0.5,
                bijector_fn=FeedforwardBijectorFunction(
                    hidden_layer_sizes=self._hidden_layer_sizes,
                    activation=self._activation),
                name=f'real_nvp_{i}')
            flow_parts += [real_nvp_bijector]

        # bijectors.Chain applies the list of bijectors in the
        # _reverse_ order of what they are inputted, thus [::-1].
        self.flow = bijectors.Chain(flow_parts[::-1])
        self._built = True

    def _forward(self, x, **condition_kwargs):
        if not self._built:
            self._build(x.shape)

        condition_kwargs = {
            bijector.name: condition_kwargs
            for bijector in self.flow.bijectors
            if isinstance(bijector, bijectors.RealNVP)
        }

        x = self.flow.forward(x, **condition_kwargs)
        return x

    def _inverse(self, y, **condition_kwargs):
        if not self._built:
            self._build(y.shape)

        condition_kwargs = {
            bijector.name: condition_kwargs
            for bijector in self.flow.bijectors
            if isinstance(bijector, bijectors.RealNVP)
        }

        y = self.flow.inverse(y, **condition_kwargs)
        return y

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        if not self._built:
            self._build(x.shape)

        condition_kwargs = {
            bijector.name: condition_kwargs
            for bijector in self.flow.bijectors
            if isinstance(bijector, bijectors.RealNVP)
        }

        fldj = self.flow.forward_log_det_jacobian(
            x, event_ndims=1, **condition_kwargs)
        return fldj

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        if not self._built:
            self._build(y.shape)

        condition_kwargs = {
            bijector.name: condition_kwargs
            for bijector in self.flow.bijectors
            if isinstance(bijector, bijectors.RealNVP)
        }

        ildj = self.flow.inverse_log_det_jacobian(
            y, event_ndims=1, **condition_kwargs)
        return ildj
