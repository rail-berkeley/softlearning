"""Tests for ConditionalRealNVPFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability import bijectors
import numpy as np

from softlearning.distributions.real_nvp_flow import ConditionalRealNVPFlow


class ConditionalRealNVPFlowTest(tf.test.TestCase):
    def test_build(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        num_coupling_layers = 10
        hidden_layer_sizes = (64, 64)

        flow = ConditionalRealNVPFlow(
            num_coupling_layers=num_coupling_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            event_dims=x_.shape[1:])

        real_nvp_layers = [
            layer for layer in flow.flow
            if isinstance(layer, bijectors.RealNVP)
        ]
        self.assertEqual(len(real_nvp_layers), num_coupling_layers)

        permute_layers = [
            layer for layer in flow.flow
            if isinstance(layer, bijectors.Permute)
        ]
        self.assertEqual(len(permute_layers), num_coupling_layers-1)

        batch_normalization_layers = [
            layer for layer in flow.flow
            if isinstance(layer, bijectors.BatchNormalization)
        ]
        self.assertEqual(len(batch_normalization_layers), 0)

        self.assertEqual(
            len(flow.flow), len(real_nvp_layers) + len(permute_layers))

    def test_forward_inverse_returns_identity(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        flow = ConditionalRealNVPFlow(
            num_coupling_layers=2,
            hidden_layer_sizes=(64,),
            event_dims=x_.shape[1:],
        )

        x = tf.constant(x_)
        forward_x = flow.forward(x)
        # Use identity to invalidate cache.
        inverse_y = flow.inverse(tf.identity(forward_x))
        forward_inverse_y = flow.forward(inverse_y)
        fldj = flow.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = flow.inverse_log_det_jacobian(tf.identity(forward_x), event_ndims=1)

        self.evaluate(tf.global_variables_initializer())

        forward_x_ = self.evaluate(forward_x)
        inverse_y_ = self.evaluate(inverse_y)
        forward_inverse_y_ = self.evaluate(forward_inverse_y)
        ildj_ = self.evaluate(ildj)
        fldj_ = self.evaluate(fldj)

        self.assertEqual("conditional_real_nvp_flow", flow.name)
        self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-4, atol=0.)
        self.assertAllClose(x_, inverse_y_, rtol=1e-4, atol=0.0)
        self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.0)

    def test_should_reuse_scale_and_log_scale_variables(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        flow = ConditionalRealNVPFlow(
            num_coupling_layers=2,
            hidden_layer_sizes=(64,),
            event_dims=x_.shape[1:],
        )

        x = tf.constant(x_)

        assert not tf.trainable_variables()

        forward_x = flow.forward(x)

        self.assertEqual(
            len(tf.trainable_variables()), 4 * flow._num_coupling_layers)

        inverse_y = flow.inverse(tf.identity(forward_x))
        forward_inverse_y = flow.forward(inverse_y)
        fldj = flow.forward_log_det_jacobian(x, event_ndims=1)
        ildj = flow.inverse_log_det_jacobian(
            tf.identity(forward_x), event_ndims=1)

        self.evaluate(tf.global_variables_initializer())

        self.assertEqual(
            len(tf.trainable_variables()), 4 * flow._num_coupling_layers)


if __name__ == '__main__':
    tf.test.main()
