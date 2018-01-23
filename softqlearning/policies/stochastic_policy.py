import tensorflow as tf
import numpy as np

from rllab.core.serializable import Serializable

from softqlearning.misc.mlp import mlp
from softqlearning.policies.nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy"""
    def __init__(self, env_spec, hidden_layer_sizes):
        Serializable.quick_init(self, locals())

        self._action_dim = env_spec.action_space.flat_dim
        self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation',
        )

        self._action_t = self.action_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            env_spec,
            self._observation_ph,
            self._action_t,
            'policy'
        )

    def action_for(self, observation, num_samples=1, reuse=tf.AUTO_REUSE):

        # TODO: should use a different latent for each sample. Currently only
        # one latent is sampled, and copied (through broadcasting) along the
        # first axis.

        n_dims = observation.get_shape().ndims
        sample_shape = np.ones(n_dims + 1, dtype=np.int32)
        sample_shape[n_dims-1:] = (num_samples, self._layer_sizes[-1])

        xi = tf.random_normal(sample_shape)  # 1 x ... x 1 x K x Do
        expanded_inputs = [tf.expand_dims(observation, n_dims-1), xi]

        with tf.variable_scope('policy', reuse=reuse):
            output = mlp(
                expanded_inputs,
                self._layer_sizes,
                nonlinearity=tf.nn.relu,
                output_nonlinearity=None)

        # Drop the sample axis if requesting only a single sample.
        if num_samples == 1:
            output = output[..., 0, :]

        return output
