import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
import numpy as np

from softlearning.distributions.squash_bijector import SquashBijector


@test_util.run_all_in_graph_and_eager_modes
class TestSquashBijector(tf.test.TestCase):
    def test_matches_tanh_bijector_single(self):
        squash = SquashBijector()
        tanh = tfp.bijectors.Tanh()
        data = np.linspace(-5, 5, 100).astype(np.float32)

        squash_forward = squash.forward(data)
        tanh_forward = tanh.forward(data)

        self.assertAllClose(
            self.evaluate(squash_forward), self.evaluate(tanh_forward))

        squash_ildj = squash.inverse_log_det_jacobian(
            squash_forward, event_ndims=0)
        tanh_ildj = tanh.inverse_log_det_jacobian(
            tanh_forward, event_ndims=0)

        tanh_finite_mask = tf.where(tf.is_finite(tanh_ildj))

        self.assertAllClose(
            self.evaluate(tf.gather(tanh_ildj, tanh_finite_mask)),
            self.evaluate(tf.gather(squash_ildj, tanh_finite_mask)),
            rtol=1e-3)

    def test_matches_tanh_bijector_double(self):
        squash = SquashBijector()
        tanh = tfp.bijectors.Tanh()
        data = np.linspace(-10, 10, 100).astype(np.float64)

        squash_forward = squash.forward(data)
        tanh_forward = tanh.forward(data)

        self.assertAllClose(
            self.evaluate(squash_forward), self.evaluate(tanh_forward))

        squash_ildj = squash.inverse_log_det_jacobian(
            squash_forward, event_ndims=0)
        tanh_ildj = tanh.inverse_log_det_jacobian(
            tanh_forward, event_ndims=0)

        self.assertAllClose(
            self.evaluate(squash_ildj), self.evaluate(tanh_ildj))


if __name__ == "__main__":
    tf.test.main()
