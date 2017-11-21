import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.parameterized import Parameterized

from sac.misc.mlp import mlp
from sac.misc import tf_utils


class ValueFunction(Parameterized, Serializable):

    def __init__(self, name, input_pls, hidden_layer_sizes):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._input_pls = input_pls
        self._layer_sizes = list(hidden_layer_sizes) + [None]

        self._output_t = self.get_output_for(*self._input_pls)

    def get_output_for(self, *inputs, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            value_t = mlp(
                inputs=inputs,
                output_nonlinearity=None,
                layer_sizes=self._layer_sizes,
            )  # N

        return value_t

    def eval(self, *inputs):
        feeds = {pl: val for pl, val in zip(self._input_pls, inputs)}

        return tf_utils.get_default_session().run(self._output_t, feeds)

    def get_params_internal(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope = tf.get_variable_scope().name
        scope += '/' + self._name + '/' if len(scope) else self._name + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )


class NNVFunction(ValueFunction):

    def __init__(self, env_spec, hidden_layer_sizes=(100, 100)):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        super(NNVFunction, self).__init__(
            'vf', (self._obs_pl,), hidden_layer_sizes)


class NNQFunction(ValueFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100)):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNQFunction, self).__init__(
            'qf', (self._obs_pl, self._action_pl), hidden_layer_sizes)
