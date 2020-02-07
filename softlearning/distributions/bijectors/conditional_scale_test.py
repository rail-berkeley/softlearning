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

    # @parameterized.named_parameters(
    #     dict(testcase_name='float32', dtype=np.float32),
    #     dict(testcase_name='float64', dtype=np.float64),
    # )
    # def testScalarCongruency(self, dtype):
    #     scale = dtype(0.42)
    #     bijector = bijectors.ConditionalScale()
    #     bijector_test_util.assert_scalar_congruency(
    #         bijector,
    #         lower_x=dtype(-2.),
    #         upper_x=dtype(2.),
    #         eval_func=self.evaluate)

    # @test_util.jax_disable_variable_test
    # def testVariableGradients(self):
    #     scale = tf.Variable(2.)
    #     b = bijectors.ConditionalScale()

    #     with tf.GradientTape() as tape:
    #         breakpoint()
    #         y = b.forward(.1, scale=scale)
    #     self.assertAllNotNone(tape.gradient(y, b.trainable_variables))

    # def testImmutableScaleAssertion(self):
    #     with self.assertRaisesOpError('Argument `scale` must be non-zero'):
    #         b = bijectors.ConditionalScale(validate_args=True)
    #         _ = self.evaluate(b.forward(1., scale=0.))

    # def testVariableScaleAssertion(self):
    #     scale = tf.Variable(0.)
    #     self.evaluate(scale.initializer)
    #     with self.assertRaisesOpError('Argument `scale` must be non-zero'):
    #         b = bijectors.ConditionalScale(validate_args=True)
    #         _ = self.evaluate(b.forward(1., scale=scale))

    # def testModifiedVariableScaleAssertion(self):
    #     scale = tf.Variable(1.)
    #     self.evaluate(scale.initializer)
    #     b = bijectors.ConditionalScale(validate_args=True)
    #     with self.assertRaisesOpError('Argument `scale` must be non-zero'):
    #         with tf.control_dependencies([scale.assign(0.)]):
    #             _ = self.evaluate(b.forward(1., scale=scale))


if __name__ == '__main__':
    tf.test.main()
