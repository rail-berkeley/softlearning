from railrl.policies.nn_policy import NNPolicy
import tensorflow as tf


class LstmPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            num_units,
            forget_bias=1.0,
            activation=tf.tanh,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(
            self._num_units,
            forget_bias=self._forget_bias,
            input_size=self.observation_dim,
            state_is_tuple=True,
            activation=self._activation
        )
