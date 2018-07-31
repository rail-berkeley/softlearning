import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc.nn import MLPFunction


class MLPPreprocessor(MLPFunction):
    def __init__(self,
                 observation_shape,
                 layer_sizes,
                 output_nonlinearity=None,
                 name='observations_preprocessor'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        assert len(observation_shape) == 1, observation_shape
        self._Do = observation_shape[0]
        self._observations_ph = tf.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')

        super(MLPPreprocessor, self).__init__(
            (self._observations_ph, ),
            name=name,
            layer_sizes=layer_sizes,
            output_nonlinearity=output_nonlinearity
        )
