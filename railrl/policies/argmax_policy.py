import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy


class ArgmaxPolicy(NeuralNetwork, Policy, Serializable):
    """
    A _policy that outputs

    pi(s) = argmax_a Q(a, s)

    The _policy is optimized using a gradient descent method on the action.
    """

    def __init__(
            self,
            name_or_scope,
            qfunction,
            learning_rate=1e-1,
            n_update_steps=25,
            **kwargs
    ):
        """

        :param name_or_scope:
        :param proposed_action: tf.Variable, which will be optimized for the
        state.
        :param qfunction: Some NNQFunction
        :param action_dim:
        :param observation_dim:
        :param learning_rate: Gradient descent learning rate.
        :param n_update_steps: How many gradient descent steps to take to
        figure out the action.
        :param kwargs:
        """
        Serializable.quick_init(self, locals())
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.n_update_steps = n_update_steps

        self.observation_input = qfunction.observation_input
        self.action_dim = qfunction.action_dim
        self.observation_dim = qfunction.observation_dim

        # Normally, this Q function is trained by getting actions. We need
        # to make a copy where the action inputted are generated from
        # internally.
        init = tf.random_uniform([1, self.action_dim],
                                 minval=0.,
                                 maxval=0.)
        with tf.variable_scope(name_or_scope) as variable_scope:
            super(ArgmaxPolicy, self).__init__(name_or_scope=variable_scope,
                                               **kwargs)
            self.proposed_action = tf.get_variable(
                name="proposed_action",
                initializer=init,
            )
            self.af_with_proposed_action = self.qfunction.get_weight_tied_copy(
                action_input=self.proposed_action
            )

            with tf.variable_scope("adam") as adam_scope:
                self.loss = -self.af_with_proposed_action.output
                self.minimizer_op = tf.train.AdamOptimizer(
                # self.minimizer_op = tf.train.GradientDescentOptimizer(
                    self.learning_rate).minimize(
                    self.loss,
                    var_list=[self.proposed_action])
                self.processed_action = tf.clip_by_value(self.proposed_action, -1, 1)
                self.adam_scope = adam_scope.name

    @overrides
    def get_params_internal(self, **tags):
        # Adam optimizer has variables as well, so don't just add
        # `super().get_params_internal()`
        return (self.qfunction.get_params_internal(**tags) +
                tf.get_collection(tf.GraphKeys.VARIABLES, self.scope_name))

    def get_action(self, observation):
        # print(observation.shape)
        # print(self.observation_dim)
        if len(observation.shape) > 1:
            assert observation.shape[1] == 1
            observation = observation.flatten()
        # Clear adam variables
        self.sess.run(tf.initialize_variables(
            tf.get_collection(tf.GraphKeys.VARIABLES, self.scope_name)
        ))
        for _ in range(self.n_update_steps):
            _, proposed, af_score =self.sess.run([
                    self.minimizer_op,
                    self.proposed_action,
                    self.af_with_proposed_action.output
                ],
                {self.observation_input: [observation]})
            # print("proposed_action = {0}".format(proposed))
            # print("af_score = {0}".format(af_score))
        action = self.sess.run(self.processed_action)
        return action, {}

    @overrides
    @property
    def output(self):
        raise Exception("{0} has no output".format(type(self)))
