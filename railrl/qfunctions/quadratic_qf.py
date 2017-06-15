import tensorflow as tf

from railrl.predictors.mlp_state_network import MlpStateNetwork
from railrl.qfunctions.nn_qfunction import NNQFunction
from railrl.core import tf_util
from railrl.qfunctions.optimizable_q_function import OptimizableQFunction


class QuadraticQF(NNQFunction, OptimizableQFunction):
    """
    Given a policy pi, represent the Q function as

        Q(s, a) = -0.5 (mu(s) - pi(s))^T P(s) (mu(s) - pi(s))

    where

        P(s) = L(s) L(s)^T

    and L(s) is a lower triangular matrix. Both L and mu are parameterized by
    feedforward neural networks.
    """
    def __init__(
            self,
            name_or_scope,
            policy,
            observation_input=None,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._policy = policy
        if observation_input is None:
            observation_input = self._policy.observation_input
        super(QuadraticQF, self).__init__(
            name_or_scope=name_or_scope,
            observation_input=observation_input,
            **kwargs
        )

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        self._L_computer = MlpStateNetwork(
            name_or_scope="L_computer",
            output_dim=self.action_dim * self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            batch_norm_config=self._batch_norm_config,
        )
        L_output = self._add_subnetwork_and_get_output(self._L_computer)
        L_output = self._process_layer(L_output)
        # L_shape = batch:dimA:dimA
        L = tf_util.vec2lower_triangle(L_output, self.action_dim)
        self.L = L

        delta = action_input - self._policy.output
        h1 = tf.expand_dims(delta, 1)  # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(h1, L)    # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(
            h1,
            h1,
            adj_y=True,  # Compute h1 * h1^T
        )                              # h1_shape = batch:1:1
        h1 = tf.squeeze(h1, [1])       # h1_shape = batch:1
        return -0.5 * h1

    @property
    def implicit_policy(self):
        return self._policy
