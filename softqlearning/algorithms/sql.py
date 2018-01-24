import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc.tensor_utils import flatten_tensor_variables
from softqlearning.misc.nn import input_bounds
from softqlearning.misc.sampler import rollouts
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel

from .rl_algorithm import RLAlgorithm


class SQL(RLAlgorithm, Serializable):
    """
    The Soft Q-Learning algorithm.

    Equations (eq. (X)) refer to the paper "Reinforcement Learning with Deep
    Energy-Based Policies", arXiv v2.
    """
    def __init__(
            self,
            base_kwargs,
            env,
            pool,
            qf,
            policy,
            plotter=None,

            qf_target_n_particles=16,
            qf_target_update_interval=1,
            qf_lr=1E-3,

            policy_lr=1E-3,

            kernel=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,

            discount=0.99,
            alpha=1,
            reward_scale=1,

            n_eval_episodes=10,
            q_plot_settings=None,
            env_plot_settings=None,

            save_full_state=False,
    ):
        """
        :param base_kwargs: Keyword arguments for OnlineAlgorithm.
        :param env: Environment object.
        :param qf_target_n_particles: Number of uniform samples used to estimate
            the soft target value for TD learning.
        :param qf_target_update_interval: How often (after how many iterations)
            the target network is updated to match the current Q-function.
        :param qf_lr: TD learning rate.
        :param policy_lr: SVGD learning rate.
        :param kernel_n_particles: Total number of particles per state used in
            the SVGD updates.
        :param kernel_update_ratio: The ratio of SVGD particles used for the
            computation of the inner/outer empirical expectation.
        :param discount: Discount factor.
        :param alpha: SVGD alpha parameter (= temperature).
        :param n_eval_episodes: Number of evaluation episodes.
        :param q_plot_settings: Settings for Q-function plots.
        :param env_plot_settings: Settings for rollout plot.
        """
        Serializable.quick_init(self, locals())
        super().__init__(**base_kwargs)

        self._env = env
        self._pool = pool
        self._qf = qf
        self._policy = policy
        self._plotter = plotter

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._alpha = alpha
        self._reward_scale = reward_scale

        self._kernel = kernel
        self._kernel_K = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        # "Fixed particles" are used to compute the inner empirical expectation
        # in the SVGD update (first kernel argument); "updated particles" are
        # used to compute the outer empirical expectation (second kernel
        # argument).
        self._kernel_K_updated = int(self._kernel_K * self._kernel_update_ratio)
        self._kernel_K_fixed = self._kernel_K - self._kernel_K_updated

        self._qf_target_K = qf_target_n_particles
        self._qf_target_update_interval = qf_target_update_interval

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._create_placeholders()
        self._create_policy()
        self._create_qf()

        self._qf_params = self._qf.get_params_internal()

        self._training_ops = []
        self._target_ops = []
        self._init_svgd_update()
        self._init_td_update()
        self._init_target_ops()

        self._q_plot_settings = q_plot_settings
        self._env_plot_settings = env_plot_settings

        self._n_eval_episodes = n_eval_episodes
        self._save_full_state = save_full_state

        self._sess.run(tf.global_variables_initializer())

    @property
    def env(self):
        return self._env

    @overrides
    def train(self):
        """ Starts the Soft Q-Learning algorithm. """
        self._train(self._env, self._policy, self._pool)

    def _create_placeholders(self):
        """ Creates all necessary placeholders. """
        # We use tf_proxy for the observation placeholder to make it
        # serializable. This is needed to make also the policy serializable.
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._obs_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='next_observation',
        )
        self._actions_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._actions_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='next_actions',
        )

        self._reward_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='rewards',
        )

        self._terminal_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminals',
        )

    def _create_policy(self):
        """
        TODO
        """

        self._policy_t = self.policy.action_for(
            observation=self._obs_pl,
            num_samples=self._kernel_K)  # N x K x Da

        self._actions_fixed, self._actions_updated = tf.split(
            self._policy_t,
            [self._kernel_K_fixed, self._kernel_K_updated],
            axis=1
        )  # N x (K_fix / K_upd) x Da

        # The gradients should not be back-propagated into the inner
        # empirical expectation.
        self._actions_fixed = tf.stop_gradient(self._actions_fixed)

        self._policy_params = self.policy.get_params_internal()

    def _create_qf(self):
        """
        Creates three Q-functions: one for the TD update, one for SVGD,
        and one for visualization. They all share the same parameters, but have
        different input/output dimensions. Additionally, the method creates a
        separate network (not sharing weights) that serves as a target network
        for the TD updates.
        """

        # Actions are normalized, and should reside between -1 and 1. The
        # environment will clip the actions, so we'll encode that as a prior
        # directly into the Q-function.
        clipped_actions = tf.clip_by_value(self._actions_pl, -1, 1)
        self._qf_t = self._qf.get_output_for(
            self._obs_pl, clipped_actions, reuse=True)  #  N x 1

        # SVGD target Q-function. Expand the dimensions to make use of
        # broadcasting (see documentation for NeuralNetwork). This will
        # evaluate the Q-function for each state-action input pair.
        obs_expanded = tf.expand_dims(self._obs_pl, axis=1)  # N x 1 x Do
        qf_unbounded_net = self._qf.get_output_for(
            obs_expanded, self._actions_fixed, reuse=True)  #  N x K_fix x 1

        # InputBounds modifies the gradient outside the action boundaries to
        # point back into the action domain. This is needed since SVGD
        # assumes unconstrained target domain, so actions may "leave" their
        # domain temporarily, but the modified gradient will eventually
        # bring them back.
        self._qf_svgd_target = input_bounds(self._actions_fixed,
                                            qf_unbounded_net)

        with tf.variable_scope('qf_td_target'):
            # Creates TD target network. Value of the next state is approximated
            # with uniform samples.
            obs_next_expanded = tf.expand_dims(self._obs_next_pl, axis=1)
            # N x 1 x Do
            target_actions = tf.random_uniform(
                (1, self._qf_target_K, self._Da), -1, 1)  # 1 x K x Da
            self._q_value_td_target = self._qf.get_output_for(
                obs_next_expanded, target_actions)  # N x 1 x 1

            self._target_qf_params = self._qf.get_params_internal()

    def _init_svgd_update(self):
        """ Creates a TF operation for the SVGD update. """

        # Target log-density. Q_soft in eq. (13):
        log_p = self._qf_svgd_target  # N x K_fix x 1

        grad_log_p = tf.gradients(log_p, self._actions_fixed)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)  # N x K_fix x 1 x Da
        grad_log_p = tf.stop_gradient(grad_log_p)

        kernel_dict = self._kernel(
            xs=self._actions_fixed,
            ys=self._actions_updated)

        kappa = tf.expand_dims(  # Kernel function in eq. (13).
            kernel_dict["output"],
            dim=3,
        )  # N x K_fix x K_upd x 1

        # Stein Variational Gradient! Eq. (13):
        action_grads = tf.reduce_mean(
            kappa * grad_log_p  # N x K_fix x K_upd x Da
            + self._alpha * kernel_dict["gradient"],
            reduction_indices=1,
        )  # N x K_upd x Da

        # Propagate the gradient through the policy network. Eq. (14):
        param_grads = tf.gradients(
            self._actions_updated,
            self._policy_params,
            grad_ys=action_grads,
        )

        # TODO: why `flatten_tensor_variables`?
        svgd_training_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=-flatten_tensor_variables(self._policy_params),
            var_list=self._policy_params,
            grad_loss=flatten_tensor_variables(param_grads)
        )

        self._training_ops.append(svgd_training_op)

    def _init_td_update(self):
        """ Creates a TF operation for the TD update. """
        if len(self._qf_params) == 0:
            return

        q_curr = self._qf_t  # N x 1
        q_curr = tf.squeeze(q_curr)  # N
        q_next = self._q_value_td_target  # N x K_td x 1
        n_target_particles = tf.cast(tf.shape(q_next)[1], tf.float32)
        # = self._qf_target_K

        # Eq. (10):
        v_next = tf.squeeze(tf.reduce_logsumexp(q_next, axis=1))  # N

        # Importance weights add just a constant to the value, which is
        # irrelevant in terms of the actual policy.
        v_next -= tf.log(n_target_particles)
        v_next += self._Da * np.log(2)

        # Qhat_soft in Eq. (11):
        ys = (self._reward_scale * tf.squeeze(self._reward_pl) +
              (1 - self._terminal_pl) * self._discount * v_next)  # N
        ys = tf.stop_gradient(ys)

        # Eq (11):
        td_loss = tf.reduce_mean(tf.square(ys - q_curr))  # 1

        td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=td_loss,
            var_list=self._qf_params,
        )

        self._training_ops.append(td_train_op)
        self._td_loss = td_loss

    def _init_target_ops(self):
        source_params = self._qf_params
        target_params = self._target_qf_params

        self._target_ops = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super()._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, itr, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)

        if itr % self._qf_target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feeds = {
            self._obs_pl: batch['observations'],
            self._actions_pl: batch['actions'],
            self._obs_next_pl: batch['next_observations'],
            self._reward_pl: batch['rewards'],
            self._terminal_pl: batch['terminals'],
        }

        return feeds

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(batch)
        qf, td_loss = self._sess.run(
            [self._qf_t, self._td_loss], feed_dict)

        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('mean-sq-bellman-error', td_loss)

        self._policy.log_diagnostics(batch)
        if self._plotter:
            self._plotter.draw()


    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            return {'epoch': epoch, 'algo': self}

        return {
            'epoch': epoch,
            'policy': self._policy,
            'qf': self._qf,
            'env': self._env,
        }

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        state = Serializable.__getstate__(self)
        state.update({
            'qf-params': self._qf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return state

    def __setstate__(self, state):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, state)
        self._qf.set_param_values(state['qf-params'])
        self._policy.set_param_values(state['policy-params'])
        self._pool.__setstate__(state['pool'])
        self._env.__setstate__(state['env'])
