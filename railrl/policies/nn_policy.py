import abc

import tensorflow as tf

from railrl.core.tf_util import he_uniform_initializer, mlp, linear
from railrl.misc.rllab_util import get_action_dim
from railrl.predictors.state_network import StateNetwork
from rllab.policies.base import Policy


class NNPolicy(StateNetwork, Policy, metaclass=abc.ABCMeta):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        action_dim = get_action_dim(**kwargs)
        # Copy dict to not affect kwargs, which is used by Serialization
        new_kwargs = dict(**kwargs)
        if "action_dim" in new_kwargs:
            new_kwargs.pop("action_dim")
        super(NNPolicy, self).__init__(name_or_scope=name_or_scope,
                                       output_dim=action_dim,
                                       **new_kwargs)

    def get_action(self, observation):
        return self.sess.run(self.output,
                             {self.observation_input: [observation]}), {}


class FeedForwardPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
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
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        with tf.variable_scope("mlp"):
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
            scope_name="output_preactivations",
        )
        with tf.variable_scope("output"):
            return self.output_nonlinearity(linear(
                observation_output,
                self.observation_hidden_sizes[-1],
                self.output_dim,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            ))
