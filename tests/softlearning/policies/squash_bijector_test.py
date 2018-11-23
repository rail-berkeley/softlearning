import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from softlearning.distributions.squash_bijector import SquashBijector
tf.enable_eager_execution()


class TestSquashBijector(unittest.TestCase):
    def test_matches_tanh_bijector_single(self):
        squash = SquashBijector()
        tanh = tfp.bijectors.Tanh()
        data = np.linspace(-5, 5, 100).astype(np.float32)

        squash_forward = squash.forward(data)
        tanh_forward = tanh.forward(data)

        np.testing.assert_equal(
            squash_forward.numpy(), tanh_forward.numpy())

        squash_ildj = squash.inverse_log_det_jacobian(
            squash_forward, event_ndims=0)
        tanh_ildj = tanh.inverse_log_det_jacobian(
            tanh_forward, event_ndims=0)

        tanh_isfinite_mask = np.where(np.isfinite(tanh_ildj))

        np.testing.assert_allclose(
            tanh_ildj.numpy()[tanh_isfinite_mask],
            squash_ildj.numpy()[tanh_isfinite_mask],
            rtol=1e-3)

    def test_matches_tanh_bijector_double(self):
        squash = SquashBijector()
        tanh = tfp.bijectors.Tanh()
        data = np.linspace(-10, 10, 100).astype(np.float64)

        squash_forward = squash.forward(data)
        tanh_forward = tanh.forward(data)

        np.testing.assert_equal(
            squash_forward.numpy(), tanh_forward.numpy())

        squash_ildj = squash.inverse_log_det_jacobian(
            squash_forward, event_ndims=0)
        tanh_ildj = tanh.inverse_log_det_jacobian(
            tanh_forward, event_ndims=0)

        np.testing.assert_allclose(
            squash_ildj.numpy(), tanh_ildj.numpy())


if __name__ == '__main__':
    unittest.main()
