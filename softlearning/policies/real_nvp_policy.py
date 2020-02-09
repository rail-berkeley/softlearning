"""RealNVPPolicy."""

from collections import OrderedDict

import tensorflow as tf
import tensorflow_probability as tfp
import tree

from softlearning.distributions.bijectors.real_nvp_flow import RealNVPFlow

from .base_policy import LatentSpacePolicy


class RealNVPPolicy(LatentSpacePolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 num_coupling_layers,
                 *args,
                 activation=tf.nn.relu,
                 use_batch_normalization=False,
                 **kwargs):
        super(RealNVPPolicy, self).__init__(*args, **kwargs)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self._output_shape),
            scale_diag=tf.ones(self._output_shape))

        self.flow_model = RealNVPFlow(
            num_coupling_layers=num_coupling_layers,
            hidden_layer_sizes=hidden_layer_sizes,
            use_batch_normalization=use_batch_normalization,
            activation=activation)

        raw_action_distribution = self.flow_model(base_distribution)

        self.base_distribution = base_distribution
        self.raw_action_distribution = raw_action_distribution
        self.action_distribution = self._action_post_processor(
            raw_action_distribution)

    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        if 0 < self._smoothing_alpha:
            raise NotImplementedError(
                "TODO(hartikainen): Smoothing alpha temporarily dropped on tf2"
                " migration. Should add it back. See:"
                " https://github.com/rail-berkeley/softlearning/blob/46374df0294b9b5f6dbe65b9471ec491a82b6944/softlearning/policies/base_policy.py#L80")

        observations = self._filter_observations(observations)

        batch_shape = tf.shape(tree.flatten(observations)[0])[:-1]
        actions = self.action_distribution.sample(
            batch_shape, bijector_kwargs={
                self.flow_model.name: {'observations': observations}
            })

        return actions

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        observations = self._filter_observations(observations)
        log_probs = self.action_distribution.log_prob(
            actions,
            bijector_kwargs={
                self.flow_model.name: {'observations': observations}
            })[..., tf.newaxis]

        return log_probs

    @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        observations = self._filter_observations(observations)
        probs = self.action_distribution.prob(
            actions,
            bijector_kwargs={
                self.flow_model.name: {'observations': observations}
            })[..., tf.newaxis]

        return probs

    def get_weights(self):
        return self.flow_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.flow_model.set_weights(*args, **kwargs)

    @property
    def trainable_weights(self):
        return self.flow_model.trainable_variables

    @property
    def non_trainable_weights(self):
        return self.flow_model.non_trainable_weights

    @tf.function(experimental_relax_shapes=True)
    def get_diagnostics(self, inputs):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        actions = self.actions(inputs)
        log_pis = self.log_probs(inputs, actions)

        return OrderedDict((
            ('entropy-mean', tf.reduce_mean(-log_pis)),
            ('entropy-std', tf.math.reduce_std(-log_pis)),

            ('actions-mean', tf.reduce_mean(actions)),
            ('actions-std', tf.math.reduce_std(actions)),
            ('actions-min', tf.reduce_min(actions)),
            ('actions-max', tf.reduce_max(actions)),
        ))
