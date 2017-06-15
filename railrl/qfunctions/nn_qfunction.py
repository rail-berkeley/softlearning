import abc
import tensorflow as tf

from railrl.predictors.state_action_network import StateActionNetwork
from railrl.core.tf_util import he_uniform_initializer, mlp, linear


class NNQFunction(StateActionNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, output_dim=1, **kwargs)


class FeedForwardCritic(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        with tf.variable_scope("observation_mlp"):
            observation_output = mlp(
                observation_input,
                self.observation_dim,
                self.observation_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
            observation_output = self._process_layer(
                observation_output,
                scope_name="observation_output"
            )
        if tf.__version__.split('.')[0] < '1':
            embedded = tf.concat(1, [observation_output, action_input])
        else:
            embedded = tf.concat([observation_output, action_input], 1)

        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp"):
            fused_output = mlp(
                embedded,
                embedded_dim,
                self.embedded_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
            fused_output = self._process_layer(fused_output)

        with tf.variable_scope("output_linear"):
            return linear(
                fused_output,
                self.embedded_hidden_sizes[-1],
                1,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            )
