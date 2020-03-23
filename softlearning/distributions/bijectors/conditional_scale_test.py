"""ConditionalScale Tests."""

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from softlearning.distributions import bijectors
from softlearning.internal import test_util


@test_util.test_all_tf_execution_regimes
class ScaleBijectorTest(test_util.TestCase, parameterized.TestCase):
    """Tests correctness of the Y = scale @ x transformation."""

    def testName(self):
        bijector = bijectors.ConditionalScale()
        self.assertStartsWith(bijector.name, 'conditional_scale')

    @parameterized.named_parameters(
        dict(testcase_name='static_float32', is_static=True, dtype=np.float32),
        dict(testcase_name='static_float64', is_static=True, dtype=np.float64),
        dict(testcase_name='dynamic_float32', is_static=False, dtype=np.float32),
        dict(testcase_name='dynamic_float64', is_static=False, dtype=np.float64),
    )
    def testNoBatchScale(self, is_static, dtype):
        scale = dtype(2.0)
        bijector = bijectors.ConditionalScale(dtype=dtype)
        x = self.maybe_static(np.array([1., 2, 3], dtype), is_static)
        self.assertAllClose([2., 4, 6], bijector.forward(x, scale=scale))
        self.assertAllClose([.5, 1, 1.5], bijector.inverse(x, scale=scale))
        self.assertAllClose(
            -np.log(2.),
            bijector.inverse_log_det_jacobian(x, scale=scale, event_ndims=0))

    @parameterized.named_parameters(
        dict(testcase_name='static_float32', is_static=True, dtype=np.float32),
        dict(testcase_name='static_float64', is_static=True, dtype=np.float64),
        dict(testcase_name='dynamic_float32', is_static=False, dtype=np.float32),
        dict(testcase_name='dynamic_float64', is_static=False, dtype=np.float64),
    )
    def testBatchScale(self, is_static, dtype):
        # Batched scale
        scale = tf.constant([2., 3.], dtype=dtype)
        bijector = bijectors.ConditionalScale(dtype=dtype)
        x = self.maybe_static(np.array([1.], dtype=dtype), is_static)
        self.assertAllClose([2., 3.], bijector.forward(x, scale=scale))
        self.assertAllClose([0.5, 1./3.], bijector.inverse(x, scale=scale))
        self.assertAllClose(
            [-np.log(2.), -np.log(3.)],
            bijector.inverse_log_det_jacobian(x, scale=scale, event_ndims=0))


if __name__ == '__main__':
    tf.test.main()
