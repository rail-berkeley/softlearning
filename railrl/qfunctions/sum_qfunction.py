import tensorflow as tf

from railrl.core.tf_util import weight_variable
from railrl.qfunctions.nn_qfunction import NNQFunction


class SumCritic(NNQFunction):
    """Just output the sum of the inputs. This is used to debug."""

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        with tf.variable_scope("actions_layer") as _:
            W_actions = weight_variable(
                (self.action_dim, 1),
                initializer=tf.constant_initializer(1.),
            )
        with tf.variable_scope("observation_layer") as _:
            W_obs = weight_variable(
                (self.observation_dim, 1),
                initializer=tf.constant_initializer(1.),
            )

        return (tf.matmul(action_input, W_actions) +
                tf.matmul(observation_input, W_obs))
