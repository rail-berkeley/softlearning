"""Implementation of GaussianMLPPolicy for educational sake."""
import numpy as np
import tensorflow as tf

from railrl.core import tf_util
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy
from rllab.spaces import Box


class GaussianMLPPolicy(Policy, Serializable):
    """A policies where p(a | s) = N(mu(s), I * std(s)^2).

    Note that the covariance matrix is diagonal.
    """
    def __init__(
            self,
            env_spec,
            mean_hidden_nonlinearity=tf.nn.relu,
            mean_hidden_sizes=(32, 32),
            std_hidden_nonlinearity=tf.nn.relu,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
    ):
        """
        :param env_spec:
        :param mean_hidden_nonlinearity: nonlinearity used for the mean hidden
                                         layers
        :param mean_hidden_sizes: list of hidden_sizes for the fully-connected hidden layers
        :param std_hidden_nonlinearity: nonlinearity used for the std hidden
                                        layers
        :param std_hidden_sizes: list of hidden_sizes for the fully-connected hidden layers
        :param min_std: whether to make sure that the std is at least some
                        threshold value, to avoid numerical issues
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)
        super(GaussianMLPPolicy, self).__init__(env_spec)

        self.env_spec = env_spec
        self.sess = tf.Session()

        # Create network
        observation_dim = self.env_spec.observation_space.flat_dim
        self.observations_input = tf.placeholder(tf.float32,
                                                 shape=[None, observation_dim])
        action_dim = self.env_spec.action_space.flat_dim
        with tf.variable_scope('mean') as _:
            mlp_mean_output = tf_util.mlp(self.observations_input,
                                          observation_dim,
                                          mean_hidden_sizes,
                                          mean_hidden_nonlinearity)
            mlp_mean_output_size = mean_hidden_sizes[-1]
            self.mean = tf_util.linear(mlp_mean_output,
                                       mlp_mean_output_size,
                                       action_dim)

        with tf.variable_scope('log_std') as _:
            mlp_std_output = tf_util.mlp(self.observations_input,
                                         observation_dim,
                                         std_hidden_sizes,
                                         std_hidden_nonlinearity)
            mlp_std_output_size = std_hidden_sizes[-1]
            self.log_std = tf_util.linear(mlp_std_output,
                                          mlp_std_output_size,
                                          action_dim)
            self.std = tf.maximum(tf.exp(self.log_std), min_std)

        self._dist = DiagonalGaussian(action_dim)

        self.actions_output = tf.placeholder(tf.float32, shape=[None, action_dim])
        z = (self.actions_output - self.mean) / self.std
        self.log_likelihood = (- tf.log(self.std**2)
                               - z**2 * 0.5
                               - tf.log(2*np.pi) * 0.5)


    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        # Add extra dimension because network is set up to take in batches.
        feed_dict = {self.observations_input: [flat_obs]}
        mean, log_std =\
                self.sess.run([self.mean, self.log_std],
                              feed_dict=feed_dict)
        rnd = np.random.normal(size=mean.shape)
        action = np.exp(log_std) * rnd + mean
        return action, dict(mean=mean, log_std=log_std)


    @overrides
    def get_params_internal(self):
        return [] # TODO(vpong)

    # TODO(vpong): not sure why I need this
    @property
    def distribution(self):
        return self._dist

