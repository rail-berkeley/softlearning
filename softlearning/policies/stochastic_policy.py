import tensorflow as tf

from serializable import Serializable

from softlearning.misc.nn import feedforward_net

from .nn_policy import NNPolicy


class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 observation_shape,
                 action_shape,
                 hidden_layer_sizes,
                 squash=True,
                 name='policy'):
        self._Serializable__initialize(locals())

        # name is set again in the superclass __init__ function, but
        # actions_for needs name to be set.
        self.name = name
        self._action_shape = (*action_shape,)
        self._observation_shape = (*observation_shape,)
        self._layer_sizes = (*hidden_layer_sizes, *action_shape)
        self._squash = squash

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation')

        self._actions = self.actions_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            name,
            observation_shape,
            action_shape,
            self._observation_ph,
            self._actions)

    def actions_for(self, observations, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (
                n_state_samples, n_action_samples, *self._action_shape)
        else:
            latent_shape = (n_state_samples, *self._action_shape)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self.name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        return tf.tanh(raw_actions) if self._squash else raw_actions
