import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.misc.mlp import MLPFunction
from sac.misc import tf_utils

class NNVFunction(MLPFunction):

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


class NNQFunction(MLPFunction):
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
