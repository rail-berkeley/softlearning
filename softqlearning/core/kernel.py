import tensorflow as tf
import numpy as np
import unittest

from rllab.core.serializable import Serializable
from softqlearning.misc.tf_proxy import SerializableTensor


class Kernel(SerializableTensor):
    """ Kernel tensor.

    The output is a tensor of shape ... x Kx x Ky, where the
    dimensions Kx and Ky are determined by the inputs xs and ys. For example,

    shape of xs: ... x Kx x D
    shape of ys: ... x Ky x D

    output shape: ... x Kx x Ky
    grad shape: ... x Kx x Ky x D

    where the leading dimensions ... needs to match. The gradient is taken with
    respect to the first input ('xs').
    """
    def __init__(self, xs, ys, kappa, grad_x):
        """
        :param xs: "Left particles."
        :param ys: "Right particles."
        :param kappa: Kernel.
        :param grad_x:  Gradient of 'kappa' with respect to 'xs'.
        """
        Serializable.quick_init(self, locals())
        self._kappa = kappa  # ... x Kx x Ky
        self._xs = xs  # ... x Kx x D
        self._ys = ys  # ... x Ky x D
        self._grad_x = grad_x  # ... x Kx x Ky x D

        super().__init__(kappa)

    @property
    def grad(self):
        """
        Returns the gradient of the Kernel matrix with respect to the first
        argument of the kernel function.
        """
        return self._grad_x  # ... x Kx x Ky x D


class AdaptiveIsotropicGaussianKernel(Kernel):
    # TODO: make this to work for any number of leading axes.
    """
    """
    def __init__(self, xs, ys, h_min=1e-3):
        """
        Gaussian kernel with dynamics bandwidth equal to
        median_distance / log(Kx).

        :param xs: Left particles.
        :param ys: Right particles.
        :param h_min: Minimum bandwidth.
        """
        Serializable.quick_init(self, locals())

        self._h_min = h_min

        Kx, D = xs.get_shape().as_list()[-2:]
        Ky, D2 = ys.get_shape().as_list()[-2:]
        assert D == D2

        leading_shape = tf.shape(xs)[:-2]

        # Compute the pairwise distances of left and right particles
        diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
        # ... x Kx x Ky x D
        dist_sq = tf.reduce_sum(diff**2, axis=-1, keep_dims=False)
        # ... x Kx x Ky

        # Get median
        input_shape = tf.concat((leading_shape, [Kx*Ky]), axis=0)
        values, _ = tf.nn.top_k(
            input=tf.reshape(dist_sq, input_shape),
            k=(Kx*Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
            sorted=True
        )  # ... x floor(Ks*Kd/2)

        medians_sq = values[..., -1]  # ... (shape) (last element is the median)

        h = medians_sq / np.log(Kx)  # ... (shape)
        h = tf.maximum(h, self._h_min)
        h = tf.stop_gradient(h)  # just in case
        h_expanded_twice = tf.expand_dims(tf.expand_dims(h, -1), -1)
        # ... x 1 x 1

        kappa = tf.exp(- dist_sq / h_expanded_twice)  # ... x Kx x Ky

        # Construct the gradient
        h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
        # ... x 1 x 1 x 1
        kappa_expanded = tf.expand_dims(kappa, -1)  # ... x Kx x Ky x 1

        kappa_grad = - 2 * diff / h_expanded_thrice * kappa_expanded
        # ... x Kx x Ky x D

        super().__init__(xs, ys, kappa, kappa_grad)

        self._db = dict()
        self._db['medians_sq'] = medians_sq


class TestAdaptiveIsotropicGaussianKernel(unittest.TestCase):
    def test_multi_axis(self):
        Kx, Ky = 3, 5
        N = 1
        M = 10
        D = 20
        h_min = 1E-3

        x_pl = tf.placeholder(
            tf.float64,
            shape=[N, M, Kx, D]
        )
        y_pl = tf.placeholder(
            tf.float64,
            shape=[N, M, Ky, D]
        )
        kernel = AdaptiveIsotropicGaussianKernel(x_pl, y_pl, h_min)

        with tf.Session() as sess:
            x = np.random.randn(N, M, Kx, D)
            y = np.random.randn(N, M, Ky, D)
            feeds = {
                x_pl: x,
                y_pl: y
            }
            medians_sq = sess.run(kernel._db['medians_sq'], feeds)
            kappa, kappa_grad = sess.run([kernel.output, kernel.grad], feeds)

            # Compute and test the median with numpy
            diff = x[..., None, :] - y[..., None, :, :]

            # noinspection PyTypeChecker
            dist_sq = np.sum(diff**2, axis=-1)
            medians_sq_np = np.median(dist_sq.reshape((N, M, -1)), axis=-1)

            np.testing.assert_almost_equal(medians_sq, medians_sq_np,
                                           decimal=5, err_msg='Wrong median.')

            # Test the kernel matrix
            h = medians_sq_np / np.log(Kx)
            h = np.maximum(h, h_min)

            kappa_np = np.exp(- dist_sq / h[..., None, None])

            np.testing.assert_almost_equal(kappa, kappa_np,
                                           decimal=5, err_msg='Wrong kernel.')

            # Test the gradient
            grad_np = - 2*diff/h[..., None, None, None]*kappa_np[..., None]

            # noinspection PyTypeChecker
            np.testing.assert_almost_equal(grad_np, kappa_grad)

    def test_trivial(self):
        """
        Test a trivial case of single particle, which can be easily
        confirmed without tensor algebra.
        """
        Kx, Ky = 2, 1  # Kx need to be at least 2
        N = 1
        D = 1
        h_min = 1E-3

        x_pl = tf.placeholder(
            tf.float64,
            shape=[N, Kx, D]
        )
        y_pl = tf.placeholder(
            tf.float64,
            shape=[N, Ky, D]
        )
        kernel = AdaptiveIsotropicGaussianKernel(x_pl, y_pl, h_min)

        with tf.Session() as sess:
            x = np.zeros((N, Kx, D))
            y = np.ones((N, Ky, D))
            feeds = {
                x_pl: x,
                y_pl: y
            }
            medians_sq = sess.run(kernel._db['medians_sq'], feeds)
            kappa, kappa_grad = sess.run([kernel.output, kernel.grad], feeds)

            # Test median.
            self.assertEqual(medians_sq[0], 1, msg='Wrong median.')

            # Test kappa.
            np.testing.assert_equal(kappa.squeeze(), np.exp(-np.log([Kx, Kx])),
                                    err_msg='Wrong kernel.')

            # Test grad_x kappa.
            grad_np = 2 * np.log(Kx) * np.exp(-np.log([Kx, Kx]))
            # noinspection PyTypeChecker
            np.testing.assert_equal(kappa_grad.squeeze(), np.squeeze(grad_np))


if __name__ == '__main__':
    unittest.main()
