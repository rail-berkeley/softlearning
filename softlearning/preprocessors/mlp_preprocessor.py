import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc.nn import (
    MLPFunction,
    TemplateFunction,
    feedforward_net_template,
)
from softlearning.misc import tf_utils


class FeedforwardNetPreprocessorV2(TemplateFunction, Serializable):
    def __init__(self, *args, name='feedforward_net_preprocessor', **kwargs):
        Serializable.quick_init(self, locals())

        super(FeedforwardNetPreprocessorV2, self).__init__(
            *args, name=name, **kwargs)

    @property
    def template_function(self):
        return feedforward_net_template


class FeedforwardNetPreprocessor(Parameterized, Serializable):
    def __init__(self,
                 input_shape,
                 layer_sizes,
                 output_nonlinearity=None,
                 name='observations_preprocessor'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._observations_ph = tf.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')

        super(FeedforwardNetPreprocessor, self).__init__(
            (self._observations_ph, ),
            name=name,
            layer_sizes=layer_sizes,
            output_nonlinearity=output_nonlinearity
        )
