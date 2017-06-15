"""
:author: Vitchyr Pong

DDPG, but use the optimal action when computing the targets. The optimal
actions must be computable for a Q function (e.g. the Q function is quadratic).
"""

import tensorflow as tf
from railrl.qfunctions.nn_qfunction import NNQFunction

from railrl.algos.ddpg import DDPG
from railrl.qfunctions.optimizable_q_function import OptimizableQFunction
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class OptimalActionTargetDDPG(DDPG):
    """
    Deep Deterministic Policy Gradient variant, where the target is chosen
    using the optimal action.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy,
            qf: (OptimizableQFunction, NNQFunction),
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :return:
        """
        super().__init__(env,
                         exploration_strategy,
                         policy,
                         qf,
                         **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.target_policy = self.policy.get_copy(
            name_or_scope=TARGET_PREFIX + self.policy.scope_name,
        )
        self.target_qf = self.qf.get_copy(
            name_or_scope=TARGET_PREFIX + self.qf.scope_name,
            action_input=self.qf.implicit_policy.output,
        )
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_qf.sess = self.sess
        self.target_policy.sess = self.sess
        self._init_qf_ops()
        self._init_policy_ops()
        self._init_target_ops()
        self.sess.run(tf.global_variables_initializer())
