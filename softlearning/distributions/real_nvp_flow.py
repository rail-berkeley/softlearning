"""RealNVP bijector flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import bijectors

import numpy as np

__all__ = [
    "ConditionalRealNVPFlow",
]


def _use_static_shape(input_tensor, ndims):
    return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


class ConditionalChain(bijectors.ConditionalBijector, bijectors.Chain):
    pass


class ConditionalRealNVPFlow(bijectors.ConditionalBijector):
    """TODO"""

    def __init__(self,
                 num_coupling_layers=2,
                 hidden_layer_sizes=(64,),
                 use_batch_normalization=False,
                 event_dims=None,
                 is_constant_jacobian=False,
                 validate_args=False,
                 name="conditional_real_nvp_flow"):
        """Instantiates the `ConditionalRealNVPFlow` normalizing flow.

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
                " for ConditionalRealNVPFlow.")
        self._use_batch_normalization = use_batch_normalization

        assert event_dims is not None, event_dims
        self._event_dims = event_dims

        self.build()

        super(ConditionalRealNVPFlow, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            is_constant_jacobian=is_constant_jacobian,
            validate_args=validate_args,
            name=name)

    def build(self):
        D = np.prod(self._event_dims)

        flow = []
        for i in range(self._num_coupling_layers):
            if self._use_batch_normalization:
                batch_normalization_bijector = bijectors.BatchNormalization()
                flow.append(batch_normalization_bijector)

            real_nvp_bijector = bijectors.RealNVP(
                num_masked=D // 2,
                shift_and_log_scale_fn=conditioned_real_nvp_template(
                    hidden_layers=self._hidden_layer_sizes,
                    # TODO: test tf.nn.relu
                    activation=tf.nn.tanh),
                name='real_nvp_{}'.format(i))

            flow.append(real_nvp_bijector)

            if i < self._num_coupling_layers - 1:
                permute_bijector = bijectors.Permute(
                    permutation=list(reversed(range(D))),
                    name='permute_{}'.format(i))
                # TODO(hartikainen): We need to force _is_constant_jacobian due
                # to the event_dim caching. See the issue filed at github:
                # https://github.com/tensorflow/probability/issues/122
                permute_bijector._is_constant_jacobian = False
                flow.append(permute_bijector)

        # Note: bijectors.Chain applies the list of bijectors in the
        # _reverse_ order of what they are inputted.
        self.flow = flow

    def _get_flow_conditions(self, **condition_kwargs):
        conditions = {
            bijector.name: condition_kwargs
            for bijector in self.flow
            if isinstance(bijector, bijectors.RealNVP)
        }

        return conditions

    def _forward(self, x, **condition_kwargs):
        conditions = self._get_flow_conditions(**condition_kwargs)
        for bijector in self.flow:
            x = bijector.forward(x, **conditions.get(bijector.name, {}))

        # TODO(hartikainen): Once tfp.bijectors.Chain supports conditioning,
        # replace the above for-loops with self.flow.forward.
        # x = self.flow.forward(x, **conditions)

        return x

    def _inverse(self, y, **condition_kwargs):
        conditions = self._get_flow_conditions(**condition_kwargs)
        for bijector in reversed(self.flow):
            y = bijector.inverse(y, **conditions.get(bijector.name, {}))

        # TODO(hartikainen): Once tfp.bijectors.Chain supports conditioning,
        # replace the above for-loops with self.flow.inverse.
        # y = self.flow.inverse(y, **conditions)

        return y

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        conditions = self._get_flow_conditions(**condition_kwargs)

        # TODO(hartikainen): Once tfp.bijectors.Chain supports conditioning,
        # replace everything below with self.flow.forward_log_det_jacobian.
        # fldj = self.flow.forward_log_det_jacobian(
        #     x, event_ndims=1, **conditions)

        fldj = tf.cast(0., dtype=x.dtype.base_dtype)
        event_ndims = self._maybe_get_static_event_ndims(
            self.forward_min_event_ndims)

        if _use_static_shape(x, event_ndims):
            event_shape = x.shape[x.shape.ndims - event_ndims:]
        else:
            event_shape = tf.shape(input=x)[tf.rank(x) - event_ndims:]
        for b in self.flow:
            fldj += b.forward_log_det_jacobian(
                x, event_ndims=event_ndims, **conditions.get(b.name, {}))
            if _use_static_shape(x, event_ndims):
                event_shape = b.forward_event_shape(event_shape)
                event_ndims = self._maybe_get_static_event_ndims(event_shape.ndims)
            else:
                event_shape = b.forward_event_shape_tensor(event_shape)
                event_ndims = tf.size(input=event_shape)
                event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
                if event_ndims_ is not None:
                    event_ndims = event_ndims_
            x = b.forward(x, **conditions.get(b.name, {}))

        return fldj

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        conditions = self._get_flow_conditions(**condition_kwargs)

        # TODO(hartikainen): Once tfp.bijectors.Chain supports conditioning,
        # replace everything below with self.flow.inverse_log_det_jacobian.
        # ildj = self.flow.inverse_log_det_jacobian(
        #     y, event_ndims=1, **conditions)

        ildj = tf.cast(0., dtype=y.dtype.base_dtype)

        event_ndims = self._maybe_get_static_event_ndims(
            self.inverse_min_event_ndims)

        if _use_static_shape(y, event_ndims):
            event_shape = y.shape[y.shape.ndims - event_ndims:]
        else:
            event_shape = tf.shape(input=y)[tf.rank(y) - event_ndims:]

        for b in reversed(self.flow):
            ildj += b.inverse_log_det_jacobian(
                y, event_ndims=event_ndims, **conditions.get(b.name, {}))

            if _use_static_shape(y, event_ndims):
                event_shape = b.inverse_event_shape(event_shape)
                event_ndims = self._maybe_get_static_event_ndims(
                    event_shape.ndims)
            else:
                event_shape = b.inverse_event_shape_tensor(event_shape)
                event_ndims = tf.size(input=event_shape)
                event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
                if event_ndims_ is not None:
                    event_ndims = event_ndims_

            y = b.inverse(y, **conditions.get(b.name, {}))

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
