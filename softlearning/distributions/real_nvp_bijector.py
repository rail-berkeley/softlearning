"""RealNVP bijector flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

__all__ = [
    "ConditionalRealNVPFlow",
]


tfb = tfp.bijectors


class ConditionalChain(tfb.ConditionalBijector, tfb.Chain):
    pass


class ConditionalRealNVPFlow(tfb.ConditionalBijector):
    """TODO"""

    def __init__(self,
                 num_coupling_layers=2,
                 hidden_layer_sizes=(64,),
                 event_ndims=1,
                 event_dims=None,
                 validate_args=False,
                 name="conditional_real_nvp_flow"):
        """Instantiates the `ConditionalRealNVPFlow` normalizing flow.

        Args:
            TODO
            event_ndims: Python scalar indicating the number of dimensions
                associated with a particular draw from the distribution.
            event_dims: Python list indicating the size of each dimension
                associated with a particular draw from the distribution.
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str` name given to ops managed by this object.

        Raises:
            ValueError: if TODO happens
        """
        assert event_ndims == 1, event_ndims
        assert event_dims is not None and len(event_dims) == 1, event_dims

        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        self._num_coupling_layers = num_coupling_layers
        self._hidden_layer_sizes = tuple(hidden_layer_sizes)

        self._event_dims = event_dims

        self.build()

        super().__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name)

    def build(self):
        D = np.prod(self._event_dims)

        flow_parts = []
        for i in range(self._num_coupling_layers):
            real_nvp_bijector = tfp.bijectors.real_nvp.RealNVP(
                num_masked=D // 2,
                shift_and_log_scale_fn=conditioned_real_nvp_template(
                    hidden_layers=self._hidden_layer_sizes,
                    # TODO: test tf.nn.relu
                    activation=tf.nn.tanh),
                name='real_nvp_{}'.format(i))

            flow_parts.append(real_nvp_bijector)

            if i < self._num_coupling_layers - 1:
                permute_bijector = tfb.Permute(
                    permutation=list(reversed(range(D))),
                    name='permute_{}'.format(i))
                flow_parts.append(permute_bijector)

        # Note: tfb.Chain applies the list of bijectors in the _reverse_ order
        # of what they are inputted.
        self.flow = ConditionalChain(list(reversed(flow_parts)))

    def _get_flow_conditions(self, **condition_kwargs):
        conditions = {
            bijector.name: condition_kwargs
            for bijector in self.flow.bijectors
            if isinstance(bijector, RealNVP)
        }

        return conditions

    def _forward(self, x, **condition_kwargs):
        self._maybe_assert_valid_x(x)

        conditions = self._get_flow_conditions(**condition_kwargs)
        out = self.flow.forward(x, **conditions)

        return out

    def _inverse(self, y, **condition_kwargs):
        self._maybe_assert_valid_y(y)

        conditions = self._get_flow_conditions(**condition_kwargs)
        out = self.flow.inverse(y, **conditions)

        return out

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        self._maybe_assert_valid_x(x)

        conditions = self._get_flow_conditions(**condition_kwargs)
        log_det_jacobian = self.flow.forward_log_det_jacobian(
            x, event_ndims=1, **conditions)

        return log_det_jacobian

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        self._maybe_assert_valid_y(y)

        conditions = self._get_flow_conditions(**condition_kwargs)
        log_det_jacobian = self.flow.inverse_log_det_jacobian(
            y, event_ndims=1, **conditions)

        return log_det_jacobian

    def _maybe_assert_valid_x(self, x):
        """TODO"""
        if not self.validate_args:
            return x
        raise NotImplementedError("_maybe_assert_valid_x")

    def _maybe_assert_valid_y(self, y):
        """TODO"""
        if not self.validate_args:
            return y
        raise NotImplementedError("_maybe_assert_valid_y")


def conditioned_real_nvp_template(hidden_layers,
                                  shift_only=False,
                                  activation=tf.nn.relu,
                                  name=None,
                                  *args,  # pylint: disable=keyword-arg-before-vararg
                                  **kwargs):

    with tf.name_scope(name, "conditioned_real_nvp_template"):

        def _fn(x, output_units, **condition_kwargs):
            """MLP which concatenates the condition kwargs to input."""
            x = tf.concat(
                (x, *[condition_kwargs[k] for k in sorted(condition_kwargs)]),
                axis=-1)

            for units in hidden_layers:
                x = tf.layers.dense(
                    inputs=x,
                    units=units,
                    activation=activation,
                    *args,  # pylint: disable=keyword-arg-before-vararg
                    **kwargs)
            x = tf.layers.dense(
                inputs=x,
                units=(1 if shift_only else 2) * output_units,
                activation=None,
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs)

            if shift_only:
              return x, None

            shift, log_scale = tf.split(x, 2, axis=-1)
            return shift, log_scale

        return tf.make_template("conditioned_real_nvp_template", _fn)
