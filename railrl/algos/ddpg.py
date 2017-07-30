"""
:author: Vitchyr Pong
"""
from collections import OrderedDict
from typing import List

import numpy as np
import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from railrl.algos.online_algorithm import OnlineAlgorithm
from railrl.policies.nn_policy import NNPolicy
from railrl.qfunctions.nn_qfunction import NNQFunction
from softqlearning.misc import logger
# from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class DDPG(OnlineAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy: NNPolicy,
            qf: NNQFunction,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            qf_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the qf
        :param policy_learning_rate: Learning rate of the _policy
        :param qf_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.qf = qf
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.qf_weight_decay = qf_weight_decay

        super().__init__(env, policy, exploration_strategy, **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.target_policy = self.policy.get_copy(
            name_or_scope=TARGET_PREFIX + self.policy.scope_name,
        )
        self.target_qf = self.qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.qf.scope_name,
            action_input=self.target_policy.output
        )
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_qf.sess = self.sess
        self.target_policy.sess = self.sess
        self._init_qf_ops()
        self._init_policy_ops()
        self._init_target_ops()
        self.sess.run(tf.global_variables_initializer())

    def _init_qf_ops(self):
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_qf.output)
        if tf.__version__.split('.')[0] < '1':
            self.qf_loss = tf.reduce_mean(
                tf.square(
                    tf.sub(self.ys, self.qf.output)))
        else:
            self.qf_loss = tf.reduce_mean(
                tf.square(
                    tf.subtract(self.ys, self.qf.output)))
        if tf.__version__.split('.')[0] < '1':
            self.Q_weights_norm = tf.reduce_sum(
                tf.pack(
                    [tf.nn.l2_loss(v)
                     for v in
                     self.qf.get_params_internal(regularizable=True)]
                ),
                name='weights_norm'
            )
        else:
            self.Q_weights_norm = tf.reduce_sum(
                tf.stack(
                    [tf.nn.l2_loss(v)
                     for v in
                     self.qf.get_params_internal(regularizable=True)]
                ),
                name='weights_norm'
            )
        self.qf_total_loss = (
            self.qf_loss + self.qf_weight_decay * self.Q_weights_norm)
        self.train_qf_op = tf.train.AdamOptimizer(
            self.qf_learning_rate).minimize(
            self.qf_total_loss,
            var_list=self.qf.get_params_internal())

    def _init_policy_ops(self):
        # To compute the surrogate loss function for the qf, it must take
        # as input the output of the _policy. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=self.policy.output)
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal())

    def _init_target_ops(self):
        policy_vars = self.policy.get_params_internal()
        qf_vars = self.qf.get_params_internal()
        target_policy_vars = self.target_policy.get_params_internal()
        target_qf_vars = self.target_qf.get_params_internal()
        assert len(policy_vars) == len(target_policy_vars)
        assert len(qf_vars) == len(target_qf_vars)

        self.update_target_policy_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_policy_vars, policy_vars)]
        self.update_target_qf_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_qf_vars, qf_vars)]

    @overrides
    def _init_training(self):
        self.target_qf.set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())

    @overrides
    @property
    def _networks(self) -> List[NeuralNetwork]:
        return [self.policy, self.qf, self.target_policy, self.target_qf]

    @overrides
    def _get_training_ops(self):
        ops = [
            self.train_policy_op,
            self.train_qf_op,
            self.update_target_qf_op,
            self.update_target_policy_op,
        ]
        if self._batch_norm:
            ops += self.qf.batch_norm_update_stats_op
            ops += self.policy.batch_norm_update_stats_op
        return ops

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        qf_feed = self._qf_feed_dict(rewards,
                                     terminals,
                                     obs,
                                     actions,
                                     next_obs)
        policy_feed = self._policy_feed_dict(obs)
        feed = qf_feed.copy()
        feed.update(policy_feed)
        return feed

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.target_qf.observation_input: next_obs,
            self.target_policy.observation_input: next_obs,
        }

    def _policy_feed_dict(self, obs):
        return {
            self.qf_with_action_input.observation_input: obs,
            self.policy.observation_input: obs,
        }

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
        (
            policy_loss,
            qf_loss,
            policy_output,
            target_policy_output,
            qf_output,
            target_qf_outputs,
            ys,
        ) = self.sess.run(
            [
                self.policy_surrogate_loss,
                self.qf_loss,
                self.policy.output,
                self.target_policy.output,
                self.qf.output,
                self.target_qf.output,
                self.ys,
            ],
            feed_dict=feed_dict)
        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths]
        returns = [sum(path["rewards"]) for path in paths]
        rewards = np.hstack([path["rewards"] for path in paths])

        # Log statistics
        last_statistics = OrderedDict([
            ('Epoch', epoch),
            ('AverageReturn', np.mean(returns)),
            ('PolicySurrogateLoss', policy_loss),
            ('QfLoss', qf_loss),
        ])
        last_statistics.update(create_stats_ordered_dict('Ys', ys))
        last_statistics.update(create_stats_ordered_dict('PolicyOutput',
                                                         policy_output))
        last_statistics.update(create_stats_ordered_dict('TargetPolicyOutput',
                                                         target_policy_output))
        last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
                                                         target_qf_outputs))
        last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        last_statistics.update(create_stats_ordered_dict('Returns', returns))
        last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                         discounted_returns))
        if len(es_path_returns) > 0:
            last_statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                             es_path_returns))
        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return last_statistics
