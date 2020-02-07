"""Scale bijector."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util


__all__ = [
    'ConditionalScale',
]


class ConditionalScale(bijector.Bijector):
    def __init__(self,
                 dtype=tf.float32,
                 validate_args=False,
                 name='conditional_scale'):
        """Instantiates the `ConditionalScale` bijector.

        This `Bijector`'s forward operation is:

        ```none
        Y = g(X) = scale * X
        ```

        Args:
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
          name: Python `str` name given to ops managed by this object.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(ConditionalScale, self).__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                validate_args=validate_args,
                dtype=dtype,
                parameters=parameters,
                name=name)

    def _maybe_assert_valid_scale(self, scale):
        if not self.validate_args:
            return ()
        is_non_zero = assert_util.assert_none_equal(
            scale,
            tf.zeros((), dtype=scale.dtype),
            message='Argument `scale` must be non-zero.')
        return (is_non_zero, )

    def _forward(self, x, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return x * scale

    def _inverse(self, y, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return y / scale

    def _forward_log_det_jacobian(self, x, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return tf.math.log(tf.abs(scale))
