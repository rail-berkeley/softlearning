import tensorflow as tf

from railrl.core import tf_util
from railrl.core.tf_util import he_uniform_initializer
from railrl.predictors.state_network import StateNetwork


class MlpStateNetwork(StateNetwork):
    def __init__(
            self,
            name_or_scope,
            output_dim,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super(MlpStateNetwork, self).__init__(name_or_scope=name_or_scope,
                                              output_dim=output_dim,
                                              **kwargs)

    def _create_network_internal(self, observation_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        with tf.variable_scope("hidden_mlp"):
            hidden_output = tf_util.mlp(
                observation_input,
                self.observation_dim,
                self.observation_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
        hidden_output = self._process_layer(hidden_output,
                                            scope_name="hidden_output")
        with tf.variable_scope("output"):
            return self.output_nonlinearity(tf_util.linear(
                hidden_output,
                self.observation_hidden_sizes[-1],
                self.output_dim,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            ))
