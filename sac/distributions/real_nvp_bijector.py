"""RealNVP bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.distributions import bijector

import tensorflow as tf
import numpy as np

__all__ = [ "RealNVPBijector", ]


def checkerboard(shape, parity="even", dtype=tf.bool):
    """TODO: Check this implementation"""
    unit = (
        tf.constant((True, False))
        if parity == "even"
        else tf.constant((False, True)))

    tiled = tf.tile(unit, (np.prod(shape) // 2,))
    return tf.cast(tf.reshape(tiled, shape), dtype)


def feedforward_net(inputs,
                    layer_sizes,
                    activation_fn=tf.nn.tanh,
                    output_nonlinearity=None,
                    regularizer=None):
    prev_size = inputs.get_shape().as_list()[-1]
    out = inputs
    for i, layer_size in enumerate(layer_sizes):
        tf.set_random_seed(seed=1)
        weight_initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.get_variable(
            name="weight_{i}".format(i=i),
            shape=(prev_size, layer_size),
            initializer=weight_initializer,
            regularizer=regularizer)

        bias_initializer = tf.initializers.random_normal()
        bias = tf.get_variable(
            name="bias_{i}".format(i=i),
            shape=(layer_size,),
            initializer=bias_initializer)

        prev_size = layer_size
        z = tf.matmul(out, weight) + bias

        if i < len(layer_sizes) - 1:
            out = activation_fn(z)
        elif output_nonlinearity is not None:
            out = output_nonlinearity(z)
        else:
            out = z

    return out


class CouplingLayer(object):
    def __init__(self, parity, name, translation_fn, scale_fn):
        self.parity = parity
        self.name = name
        self.translation_fn = translation_fn
        self.scale_fn = scale_fn

    def get_mask(self, x, dtype):
        shape = x.get_shape()
        mask = checkerboard(shape[1:], parity=self.parity, dtype=dtype)

        # TODO: remove assert
        assert mask.get_shape() == shape[1:]

        return mask

    def forward_and_jacobian(self, inputs):
        with tf.variable_scope(self.name):
            shape = inputs.get_shape()
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            # masked half of inputs
            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale", reuse=tf.AUTO_REUSE):
                scale = mask * self.scale_fn(masked_inputs)

            with tf.variable_scope("translation", reuse=tf.AUTO_REUSE):
                translation = mask * self.translation_fn(masked_inputs)

            # TODO: check the masks
            exp_scale = tf.check_numerics(
                tf.exp(scale), "tf.exp(scale) contains NaNs or Infs.")
            # (9) in paper

            if self.parity == "odd":
                outputs = tf.stack((
                    inputs[:, 0] * exp_scale[:, 1] + translation[:, 1],
                    inputs[:, 1],
                ), axis=1)
            else:
                outputs = tf.stack((
                    inputs[:, 0],
                    inputs[:, 1] * exp_scale[:, 0] + translation[:, 0],
                ), axis=1)

            log_det_jacobian = tf.reduce_sum(
                scale, axis=tuple(range(1, len(shape))))

            return outputs, log_det_jacobian

    def backward_and_jacobian(self, inputs):
        """Calculate inverse of the layer

        Note that `inputs` correspond to the `outputs` in forward function
        """
        with tf.variable_scope(self.name):
            shape = inputs.get_shape()
            mask = self.get_mask(inputs, dtype=inputs.dtype)

            masked_inputs = inputs * mask

            # TODO: scale and translation could be merged into a single network
            with tf.variable_scope("scale", reuse=tf.AUTO_REUSE):
                scale = mask * self.scale_fn(masked_inputs)

            with tf.variable_scope("translation", reuse=tf.AUTO_REUSE):
                translation = mask * self.translation_fn(masked_inputs)

            if self.parity == "odd":
                outputs = tf.stack((
                    (inputs[:, 0] - translation[:, 1]) * tf.exp(-scale[:, 1]),
                    inputs[:, 1],
                ), axis=1)
            else:
                outputs = tf.stack((
                    inputs[:, 0],
                    (inputs[:, 1] - translation[:, 0]) * tf.exp(-scale[:, 0]),
                ), axis=1)

            # TODO: Should this be - or +?
            log_det_jacobian = tf.reduce_sum(
                -scale, axis=tuple(range(1, len(shape))))

            return outputs, log_det_jacobian


DEFAULT_CONFIG = {
    "num_coupling_layers": 2,
    "translation_hidden_sizes": (25,),
    "scale_hidden_sizes": (25,),
    "scale_regularization": 5e2
}

class RealNVPBijector(bijector.Bijector):
    """TODO"""

    def __init__(self,
                 config=None,
                 event_ndims=0,
                 validate_args=False,
                 name="real_nvp"):
        """Instantiates the `RealNVPBijector` bijector.

        Args:
            TODO
            event_ndims: Python scalar indicating the number of dimensions associated
                with a particular draw from the distribution.
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str` name given to ops managed by this object.

        Raises:
            ValueError: if TODO happens
        """
        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        self.config = dict(DEFAULT_CONFIG, **(config or {}))

        self.build()

        super().__init__(event_ndims=event_ndims,
                         validate_args=validate_args,
                         name=name)

    # TODO: Properties

    def build(self):
        num_coupling_layers = self.config["num_coupling_layers"]
        translation_hidden_sizes = self.config["translation_hidden_sizes"]
        scale_hidden_sizes = self.config["scale_hidden_sizes"]

        def translation_wrapper(inputs):
            return feedforward_net(
                inputs,
                # TODO: should allow multi_dimensional inputs/outputs
                layer_sizes=(*translation_hidden_sizes, inputs.shape.as_list()[-1]))

        def scale_wrapper(inputs):
            return feedforward_net(
                inputs,
                # TODO: should allow multi_dimensional inputs/outputs
                layer_sizes=(*scale_hidden_sizes, inputs.shape.as_list()[-1]),
                regularizer=tf.contrib.layers.l2_regularizer(
                    self.config["scale_regularization"]))

        self.layers = [
            CouplingLayer(
                parity=("even", "odd")[i % 2],
                name="coupling_{i}".format(i=i),
                translation_fn=translation_wrapper,
                scale_fn=scale_wrapper)
            for i in range(1, num_coupling_layers + 1)
        ]

    def _forward(self, x):
        x = self._maybe_assert_valid_x(x)

        out = x
        for layer in self.layers:
            out, _ = layer.forward_and_jacobian(out)

        return out

    def _forward_log_det_jacobian(self, x):
        x = self._maybe_assert_valid_x(x)

        sum_log_det_jacobians = tf.reduce_sum(
            tf.zeros_like(x), axis=tuple(range(1, len(x.shape))))

        out = x
        for layer in self.layers:
            out, log_det_jacobian = layer.forward_and_jacobian(out)
            assert (sum_log_det_jacobians.shape.as_list()
                    == log_det_jacobian.shape.as_list())

            sum_log_det_jacobians += log_det_jacobian

        return sum_log_det_jacobians

    def _inverse(self, y):
        y = self._maybe_assert_valid_y(y)

        out = y
        for layer in reversed(self.layers):
            out, _ = layer.backward_and_jacobian(out)

        return out

    def _inverse_log_det_jacobian(self, y):
        y = self._maybe_assert_valid_y(y)

        sum_log_det_jacobians = tf.reduce_sum(
            tf.zeros_like(y), axis=tuple(range(1, len(y.shape))))

        out = y
        for layer in reversed(self.layers):
            out, log_det_jacobian = layer.backward_and_jacobian(out)
            assert (sum_log_det_jacobians.shape.as_list()
                    == log_det_jacobian.shape.as_list())

            sum_log_det_jacobians += log_det_jacobian

        return sum_log_det_jacobians

    def _maybe_assert_valid_x(self, x):
        """TODO"""
        if not self.validate_args:
            return x
        return x

    def _maybe_assert_valid_y(self, y):
        """TODO"""
        if not self.validate_args:
            return y
        return y
