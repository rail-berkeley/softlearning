"""ConditionalShift Tests."""

# Dependency imports

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from softlearning.distributions import bijectors
from softlearning.internal import test_util


@test_util.test_all_tf_execution_regimes
class ShiftTest(test_util.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='static', is_static=True),
        dict(testcase_name='dynamic', is_static=False),
    )
    def testNoBatch(self, is_static):
        shift = bijectors.ConditionalShift()
        x = self.maybe_static([1., 1.], is_static)
        self.assertAllClose([2., 0.], shift.forward(x, shift=[1., -1.]))
        self.assertAllClose([0., 2.], shift.inverse(x, shift=[1., -1.]))
        self.assertAllClose(
            0., shift.inverse_log_det_jacobian(x, shift=[[2., -.5], [1., -3.]], event_ndims=1))

    @parameterized.named_parameters(
        dict(testcase_name='static', is_static=True),
        dict(testcase_name='dynamic', is_static=False),
    )
    def testBatch(self, is_static):
        shift = bijectors.ConditionalShift()
        x = self.maybe_static([1., 1.], is_static)

        self.assertAllClose([[3., .5], [2., -2.]], shift.forward(
            x, shift=[[2., -.5], [1., -3.]]))
        self.assertAllClose([[-1., 1.5], [0., 4.]], shift.inverse(
            x, shift=[[2., -.5], [1., -3.]]))
        self.assertAllClose(0., shift.inverse_log_det_jacobian(
            x, shift=[[2., -.5], [1., -3.]], event_ndims=1))


if __name__ == '__main__':
    tf.test.main()
