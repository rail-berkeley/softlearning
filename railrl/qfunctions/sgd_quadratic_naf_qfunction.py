import tensorflow as tf

from railrl.policies.argmax_policy import ArgmaxPolicy
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.predictors.mlp_state_network import MlpStateNetwork
from railrl.qfunctions.naf_qfunction import NAFQFunction
from railrl.qfunctions.quadratic_qf import QuadraticQF
from rllab.misc.overrides import overrides


class SgdQuadraticNAF(NAFQFunction):
    def __init__(self, name_or_scope, **kwargs):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self._implicit_policy = None

    """
    Same as QuadraticNAF except that the implicit policy is computed by doing
    SGD on the put. Used for debugging.
    """
    @overrides
    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        self._vf = MlpStateNetwork(
            name_or_scope="V_function",
            output_dim=1,
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
        self._policy = FeedForwardPolicy(
            name_or_scope="implict_policy",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            batch_norm_config=self._batch_norm_config,
        )
        self._af = QuadraticQF(
            name_or_scope="advantage_function",
            action_input=action_input,
            observation_input=observation_input,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            policy=self._policy,
            batch_norm_config=self._batch_norm_config,
        )
        vf_out = self._add_subnetwork_and_get_output(self._vf)
        af_out = self._add_subnetwork_and_get_output(self._af)
        return vf_out + af_out

    @property
    def implicit_policy(self):
        if self._implicit_policy is None:
            with self.sess.as_default():
                self.sess.run(tf.initialize_variables(self.get_params()))
                print("Making SGD optimizer")
                self._implicit_policy = ArgmaxPolicy(
                    name_or_scope="argmax_policy",
                    qfunction=self,
                    batch_norm_config=self._batch_norm_config,
                )
        return self._implicit_policy

    @property
    def value_function(self):
        return self._vf

    @property
    def advantage_function(self):
        return self._af

    @property
    def update_weights_ops(self):
        return None
