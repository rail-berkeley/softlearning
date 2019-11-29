"""RealNVP bijector flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import bijectors

import numpy as np

__all__ = [
    "RealNVPFlow",
]


def _use_static_shape(input_tensor, ndims):
    return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


class RealNVPFlow(bijectors.Bijector):
    """TODO"""

    def __init__(self,
                 num_coupling_layers=2,
                 hidden_layer_sizes=(64,),
                 use_batch_normalization=False,
                 event_dims=None,
                 is_constant_jacobian=False,
                 validate_args=False,
                 name="_real_nvp_flow"):
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

        Raises:
            ValueError: if TODO happens
        """
        self._graph_parents = []
        self._name = name

        self._num_coupling_layers = num_coupling_layers
        self._hidden_layer_sizes = tuple(hidden_layer_sizes)
        if use_batch_normalization:
            raise NotImplementedError(
                "TODO(hartikainen): Batch normalization is not yet supported"
                " for RealNVPFlow.")
        self._use_batch_normalization = use_batch_normalization

        assert event_dims is not None, event_dims
        self._event_dims = event_dims

        self.build()

        super(RealNVPFlow, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            is_constant_jacobian=is_constant_jacobian,
            validate_args=validate_args,
            name=name)

    def build(self):
        D = np.prod(self._event_dims)

        flow_parts = []
        for i in range(self._num_coupling_layers):
            if self._use_batch_normalization:
                batch_normalization_bijector = bijectors.BatchNormalization()
                flow_parts += [batch_normalization_bijector]

            real_nvp_bijector = bijectors.RealNVP(
                num_masked=D // 2,
                shift_and_log_scale_fn=conditioned_real_nvp_template(
                    hidden_layers=self._hidden_layer_sizes,
                    # TODO: test tf.nn.relu
                    activation=tf.nn.tanh),
                name='real_nvp_{}'.format(i))
            flow_parts += [real_nvp_bijector]

            if i < self._num_coupling_layers - 1:
                permute_bijector = bijectors.Permute(
                    permutation=list(reversed(range(D))),
                    name='permute_{}'.format(i))
                # TODO(hartikainen): We need to force _is_constant_jacobian due
                # to the event_dim caching. See the issue filed at github:
                # https://github.com/tensorflow/probability/issues/122
                permute_bijector._is_constant_jacobian = False
                flow_parts += [permute_bijector]

        # bijectors.Chain applies the list of bijectors in the
        # _reverse_ order of what they are inputted, thus [::-1].
        self.flow = bijectors.Chain(flow_parts[::-1])

    def _forward(self, x, **condition_kwargs):
        x = self.flow.forward(x, **condition_kwargs)
        return x

    def _inverse(self, y, **condition_kwargs):
        y = self.flow.inverse(y, **condition_kwargs)
        return y

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        fldj = self.flow.forward_log_det_jacobian(
            x, event_ndims=1, **condition_kwargs)
        return fldj

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        ildj = self.flow.inverse_log_det_jacobian(
            y, event_ndims=1, **condition_kwargs)
        return ildj


def conditioned_real_nvp_template(hidden_layers,
                                  shift_only=False,
                                  activation=tf.nn.relu,
                                  name=None,
                                  *args,  # pylint: disable=keyword-arg-before-vararg
                                  **kwargs):

    with tf.compat.v1.name_scope(name, "conditioned_real_nvp_template"):

        def _fn(x, output_units, **condition_kwargs):
            """MLP which concatenates the condition kwargs to input."""
            x = tf.concat(
                (x, *[condition_kwargs[k] for k in sorted(condition_kwargs)]),
                axis=-1)

            for units in hidden_layers:
                x = tf.compat.v1.layers.dense(
                    inputs=x,
                    units=units,
                    activation=activation,
                    *args,  # pylint: disable=keyword-arg-before-vararg
                    **kwargs)
            x = tf.compat.v1.layers.dense(
                inputs=x,
                units=(1 if shift_only else 2) * output_units,
                activation=None,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs)

            if shift_only:
                return x, None

            shift, log_scale = tf.split(x, 2, axis=-1)
            return shift, log_scale

        return tf.compat.v1.make_template("conditioned_real_nvp_template", _fn)
