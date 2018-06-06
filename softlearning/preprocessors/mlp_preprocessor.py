import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc.nn import MLPFunction
from softlearning.misc import tf_utils

class MLPPreprocessor(MLPFunction):
    def __init__(self,
                 env_spec,
                 layer_sizes,
                 output_nonlinearity=None,
                 name='observations_preprocessor'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._observations_ph = tf.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')

        super(MLPPreprocessor, self).__init__(
            (self._observations_ph, ),
            name=name,
            layer_sizes=layer_sizes,
            output_nonlinearity=output_nonlinearity
        )
