import tensorflow as tf
import tensorflow_probability as tfp
import tree

from .base_policy import ContinuousPolicy


class UniformPolicyMixin:
    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        batch_shape = tf.shape(tree.flatten(observations)[0])[:-1]
        actions = self.distribution.sample(batch_shape)

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        log_probs = self.distribution.log_prob(actions)
        return log_probs


class ContinuousUniformPolicy(UniformPolicyMixin, ContinuousPolicy):
    def __init__(self, *args, **kwargs):
        super(ContinuousUniformPolicy, self).__init__(*args, **kwargs)
        low, high = self._action_range
        self.distribution = tfp.distributions.Uniform(low=low, high=high)
