"""Tests for RealNVPFlow."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

from softlearning.distributions.bijectors.real_nvp_flow import RealNVPFlow


@pytest.mark.skip(reason="tf2 broke these tests.")
class RealNVPFlowTest(tf.test.TestCase):
    def test_build(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        num_coupling_layers = 10
        hidden_layer_sizes = (64, 64)

        flow = RealNVPFlow(
            num_coupling_layers=num_coupling_layers,
            hidden_layer_sizes=hidden_layer_sizes)

        self.assertFalse(flow._built)
        flow.forward(x_)
        self.assertTrue(flow._built)

        real_nvp_layers = [
            layer for layer in flow.flow.bijectors
            if isinstance(layer, bijectors.RealNVP)
        ]
        self.assertEqual(len(real_nvp_layers), num_coupling_layers)

        permute_layers = [
            layer for layer in flow.flow.bijectors
            if isinstance(layer, bijectors.Permute)
        ]
        self.assertEqual(len(permute_layers), num_coupling_layers-1)

        batch_normalization_layers = [
            layer for layer in flow.flow.bijectors
            if isinstance(layer, bijectors.BatchNormalization)
        ]
        self.assertEqual(len(batch_normalization_layers), 0)

        self.assertEqual(
            len(flow.flow.bijectors),
            len(real_nvp_layers) + len(permute_layers))

    def test_forward_inverse_returns_identity(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        flow = RealNVPFlow(
            num_coupling_layers=2,
            hidden_layer_sizes=(64,))

        x = tf.constant(x_)
        forward_x = flow.forward(x)
        # Use identity to invalidate cache.
        inverse_y = flow.inverse(tf.identity(forward_x))
        forward_inverse_y = flow.forward(inverse_y)
        fldj = flow.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = flow.inverse_log_det_jacobian(tf.identity(forward_x), event_ndims=1)

        forward_x_ = forward_x.numpy()
        inverse_y_ = inverse_y.numpy()
        forward_inverse_y_ = forward_inverse_y.numpy()
        ildj_ = ildj.numpy()
        fldj_ = fldj.numpy()

        self.assertEqual("real_nvp_flow", flow.name)
        self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-4, atol=0.)
        self.assertAllClose(x_, inverse_y_, rtol=1e-4, atol=0.0)
        self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.0)

    def test_should_reuse_scale_and_log_scale_variables(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        flow = RealNVPFlow(
            num_coupling_layers=2,
            hidden_layer_sizes=(64,))

        x = tf.constant(x_)

        assert not tf.compat.v1.trainable_variables()

        forward_x = flow.forward(x)

        self.assertEqual(
            len(tf.compat.v1.trainable_variables()), 4 * flow._num_coupling_layers)

        inverse_y = flow.inverse(tf.identity(forward_x))
        forward_inverse_y = flow.forward(inverse_y)
        fldj = flow.forward_log_det_jacobian(x, event_ndims=1)
        ildj = flow.inverse_log_det_jacobian(
            tf.identity(forward_x), event_ndims=1)

        self.assertEqual(
            len(tf.compat.v1.trainable_variables()), 4 * flow._num_coupling_layers)

    def test_batched_flow_with_mlp_transform(self):
        x_ = np.random.normal(0., 1., (3, 8)).astype(np.float32)
        flow = RealNVPFlow(
            num_coupling_layers=2,
            hidden_layer_sizes=(64,),
            use_batch_normalization=False)
        x = tf.constant(x_)
        forward_x = flow.forward(x)
        # Use identity to invalidate cache.
        inverse_y = flow.inverse(forward_x)
        forward_inverse_y = flow.forward(inverse_y)
        fldj = flow.forward_log_det_jacobian(x, event_ndims=1)
        # Use identity to invalidate cache.
        ildj = flow.inverse_log_det_jacobian(forward_x, event_ndims=1)

        [
            forward_x_,
            inverse_y_,
            forward_inverse_y_,
            ildj_,
            fldj_,
        ] = [
            forward_x.numpy(),
            inverse_y.numpy(),
            forward_inverse_y.numpy(),
            ildj.numpy(),
            fldj.numpy(),
        ]

        self.assertEqual("real_nvp_flow", flow.name)
        self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-4, atol=0.)
        self.assertAllClose(x_, inverse_y_, rtol=1e-4, atol=0.)
        self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=1e-8)

    def test_with_batch_normalization(self):
        x_ = np.reshape(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (-1, 4))

        with self.assertRaises(NotImplementedError):
            flow = RealNVPFlow(
                num_coupling_layers=2,
                hidden_layer_sizes=(64,),
                use_batch_normalization=True)


if __name__ == '__main__':
    tf.test.main()
