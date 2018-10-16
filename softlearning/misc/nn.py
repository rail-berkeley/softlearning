from abc import ABCMeta, abstractmethod

import tensorflow as tf

from serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized

from softlearning.misc import tf_utils


def feedforward_net_v2(inputs,
                       hidden_layer_sizes,
                       output_size,
                       activation=tf.nn.relu,
                       output_activation=None,
                       *args,
                       **kwargs):
    out = inputs
    for units in hidden_layer_sizes:
        out = tf.layers.dense(
            inputs=out,
            units=units,
            activation=activation,
            *args,
            **kwargs)

    out = tf.layers.dense(
        inputs=out,
        units=output_size,
        activation=output_activation,
        *args,
        **kwargs)

    return out


def feedforward_net_template(
        hidden_layer_sizes,
        output_size,
        activation=tf.nn.relu,
        output_activation=None,
        name="feedforward_net_template",
        create_scope_now_=False,
        *args,
        **kwargs):
    def _fn(inputs):
        return feedforward_net_v2(
            inputs,
            hidden_layer_sizes,
            output_size,
            activation=tf.nn.relu,
            output_activation=None,
            *args,
            **kwargs)

    return tf.make_template(name, _fn, create_scope_now_=create_scope_now_)


class TemplateFunction(Parameterized, Serializable, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        Parameterized.__init__(self)
        self._Serializable__initialize(locals())

        self._function = self.template_function(*args, **kwargs)

    @property
    @abstractmethod
    def template_function(self):
        pass

    def __call__(self, *inputs):
        return self._function(tf.concat(inputs, axis=-1))

    def get_params_internal(self):
        return self._function.trainable_variables


class FeedforwardFunction(TemplateFunction):
    def __init__(self, *args, name='feedforward_function', **kwargs):
        self._Serializable__initialize(locals())

        super(FeedforwardFunction, self).__init__(*args, name=name, **kwargs)

    @property
    def template_function(self):
        return feedforward_net_template


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
    def __init__(self, inputs, name, layer_sizes, output_nonlinearity=None):
        Parameterized.__init__(self)
        self._Serializable__initialize(locals())

        self._name = name
        self._inputs = inputs
        self._layer_sizes = list(layer_sizes)
        self._output_nonlinearity = output_nonlinearity

        self._output = self.output_for(*self._inputs)

    def output_for(self, *inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            out = feedforward_net(
                inputs=inputs,
                output_nonlinearity=self._output_nonlinearity,
                layer_sizes=self._layer_sizes)

        return out

    def eval(self, *inputs):
        feeds = {ph: val for ph, val in zip(self._inputs, inputs)}

        return tf_utils.get_default_session().run(self._output, feeds)

    def get_params_internal(self, scope='', **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope = scope or tf.get_variable_scope().name
        scope += '/' + self._name if len(scope) else self._name

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
