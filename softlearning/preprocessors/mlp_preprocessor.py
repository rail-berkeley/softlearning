import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc.nn import (
    MLPFunction,
    TemplateFunction,
    feedforward_net_template,
    feedforward_net_v2,
)
from softlearning.misc import tf_utils


def feedforward_net_preprocessor_template(
        hidden_layer_sizes,
        output_size,
        ignore_input=0,
        activation=tf.nn.relu,
        output_activation=None,
        name="feedforward_net_preprocessor_template",
        create_scope_now_=False,
        *args,
        **kwargs):
    def _fn(inputs):
        if ignore_input > 0:
            inputs_to_preprocess = inputs[..., :-ignore_input]
            passthrough = inputs[..., -ignore_input:]
        else:
            inputs_to_preprocess = inputs
            passthrough = inputs[..., 0:0]

        preprocessed = feedforward_net_v2(
            inputs_to_preprocess,
            hidden_layer_sizes,
            output_size-ignore_input,
            activation=tf.nn.relu,
            output_activation=None,
            *args,
            **kwargs)

        return tf.concat([preprocessed, passthrough], axis=-1)

    return tf.make_template(name, _fn, create_scope_now_=create_scope_now_)


class FeedforwardNetPreprocessorV2(TemplateFunction, Serializable):
    def __init__(self, *args, name='feedforward_net_preprocessor', **kwargs):
        Serializable.quick_init(self, locals())

        super(FeedforwardNetPreprocessorV2, self).__init__(
            *args, name=name, **kwargs)

    @property
    def template_function(self):
        return feedforward_net_preprocessor_template


class FeedforwardNetPreprocessor(Parameterized, Serializable):
    def __init__(self,
                 input_shape,
                 layer_sizes,
                 output_nonlinearity=None,
                 name='observations_preprocessor'):

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        assert len(input_shape) == 1, input_shape
        self._Do = input_shape[0]
        self._observations_ph = tf.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')

        super(FeedforwardNetPreprocessor, self).__init__(
            (self._observations_ph, ),
            name=name,
            layer_sizes=layer_sizes,
            output_nonlinearity=output_nonlinearity
        )
