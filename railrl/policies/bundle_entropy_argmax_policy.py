import numpy as np
import tensorflow as tf

from railrl.optimizers.bundle_entropy import solveBatch
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy


class BundleEntropyArgmaxPolicy(Policy, Serializable):
    """
    A policy that outputs

    pi(s) = argmax_a Q(a, s)

    The policy is optimized using the bundle entropy method described in [1].

    References
    ----------
    .. [1] Amos, Brandon, Lei Xu, and J. Zico Kolter.
           "Input Convex Neural Networks." arXiv preprint arXiv:1609.07152 (2016).
    """

    def __init__(self, qfunction, action_dim, sess, n_update_steps=10, *args,
                 **kwargs):
        """

        :param qfunction: Some NNQFunction
        :param action_dim:
        :param sess: tf.Session
        :param n_update_steps: How many optimization steps to take to figure out
        the action.
        """
        super().__init__(env_spec=None)
        Serializable.quick_init(self, locals())
        self.qfunction = qfunction
        self.action_dim = qfunction.action_dim
        self.sess = sess
        self.n_update_steps = n_update_steps
        self.loss = -qfunction.output
        self.dloss_daction = tf.gradients(
            self.loss,
            qfunction.action_input,
        )[0]

        self.observation_input = qfunction.observation_input
        self.observation_dim = qfunction.observation_dim

    def get_action(self, observation):
        debug_dict = {}
        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, axis=0)

        batch_size = observation.shape[0]

        def q_value_and_gradient(action_proposed):
            assert np.max(action_proposed) <= 1.
            assert np.min(action_proposed) >= -1.
            value, grad = self.sess.run(
                [
                    self.loss,
                    self.dloss_daction,
                ],
                feed_dict={
                    self.qfunction.action_input: action_proposed,
                    self.qfunction.observation_input: observation,
                }
            )
            value = np.squeeze(value, 1)  # make the shape = (batchsize,)
            grad *= 2  # Since the action is multiplied by two
            return value, grad

        seed_action = np.ones((batch_size, self.action_dim)) * 0.5
        raw_actions = solveBatch(q_value_and_gradient, seed_action,
                                 nIter=self.n_update_steps)[0]
        unflattened_actions = np.expand_dims(raw_actions[0], axis=1)
        # act is scaled to be only between 0 and 1.
        # clip to be super safe
        action = np.clip(2 * unflattened_actions - 1, -1, 1)
        return action, debug_dict

    @overrides
    def get_params_internal(self):
        return []
