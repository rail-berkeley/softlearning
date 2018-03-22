import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized

from softqlearning.misc import tf_utils


def feedforward_net(inputs,
                    layer_sizes,
                    activation_fn=tf.nn.relu,
                    output_nonlinearity=None):
    def bias(n_units):
        return tf.get_variable(
            name='bias', shape=n_units, initializer=tf.zeros_initializer())

    def linear(x, n_units, postfix=None):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        # `tf.tensordot` supports broadcasting
        return tf.tensordot(x, weight, axes=((-1, ), (0, )))

    out = 0
    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('layer_{i}'.format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    out += linear(input_tensor, layer_size, j)
            else:
                out = linear(out, layer_size)

            out += bias(layer_size)

            if i < len(layer_sizes) - 1 and activation_fn:
                out = activation_fn(out)

    if output_nonlinearity:
        out = output_nonlinearity(out)

    return out


class MLPFunction(Parameterized, Serializable):
    def __init__(self, inputs, name, hidden_layer_sizes):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._inputs = inputs
        self._layer_sizes = list(hidden_layer_sizes) + [1]

        self._output = self._output_for(self._inputs)

    def _output_for(self, inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            out = feedforward_net(
                inputs=inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes)

        return out[..., 0]

    def _eval(self, inputs):
        feeds = {pl: val for pl, val in zip(self._inputs, inputs)}

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope += '/' + self._name if scope else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
