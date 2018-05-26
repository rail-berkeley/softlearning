import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc.nn import MLPFunction
from softlearning.misc import tf_utils

class MLPPreprocessor(MLPFunction):
    def __init__(self, env_spec, layer_sizes=(128, 16),
                 output_nonlinearity=None, name='observations_preprocessor'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name

        self._Do = env_spec.observation_space.flat_dim

        obs_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observations',
        )

        self._inputs = (obs_ph, )
        self._layer_sizes = layer_sizes
        self._output_nonlinearity = output_nonlinearity

        self._output_t = self.output_for(obs_ph, reuse=tf.AUTO_REUSE)

    def output_for(self, observations, reuse=False):
        return super(MLPPreprocessor, self)._output_for(
            (observations, ),
            reuse=reuse)[..., None]
