from __future__ import absolute_import

import gc
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import softqlearning.misc.tf_proxy as tp

from rllab.core.serializable import Serializable
from softqlearning.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import special
from softqlearning.misc.tensor_utils import flatten_tensor_variables
from softqlearning.algos.online_algorithm import OnlineAlgorithm
from softqlearning.core.kernel import AdaptiveIsotropicGaussianKernel
from softqlearning.core.nn import InputBounds
from softqlearning.core.nn import NeuralNetwork, StochasticNeuralNetwork
from softqlearning.misc.sampler import rollouts
from softqlearning.policies.nn_policy import NNPolicy
from softqlearning.q_functions.nn_qf import NNQFunction


class SoftQLearning(OnlineAlgorithm, Serializable):
    """
    The Soft Q-Learning algorithm.
    """
    def __init__(
            self,
            base_kwargs,
            policy_kwargs,
            qf_kwargs,
            env,

            qf_class=NeuralNetwork,
            qf_target_n_particles=16,
            qf_target_update_interval=1,
            qf_lr=1E-3,

            policy_class=StochasticNeuralNetwork,
            policy_lr=1E-3,

            kernel_class=AdaptiveIsotropicGaussianKernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,

            discount=0.99,
            alpha=1,
            alpha_scheduler=None,

            eval_n_episodes=10,
            eval_render=False,
            q_plot_settings=None,
            env_plot_settings=None,
    ):
        """
        :param base_kwargs: Keyword arguments for OnlineAlgorithm.
        :param policy_kwargs: Keyword arguments for the policy class.
        :param qf_kwargs: Keyword arguments for the Q-function class.
        :param env: Environment object.
        :param qf_class: Q-function class type.
        :param qf_target_n_particles: Number of uniform samples used to estimate
            the soft target value for TD learning.
        :param qf_target_update_interval: How often (after how many iterations)
            the target network is updated to match the current Q-function.
        :param qf_lr: TD learning rate.
        :param policy_class: Policy class type.
        :param policy_lr: SVGD learning rate.
        :param kernel_class: Kernel class type.
        :param kernel_n_particles: Total number of particles per state used in
            the SVGD updates.
        :param kernel_update_ratio: The ratio of SVGD particles used for the
            computation of the inner/outer empirical expectation.
        :param discount: Discount factor.
        :param alpha: SVGD alpha parameter (= temperature).
        :param eval_n_episodes: Number of evaluation episodes.
        :param q_plot_settings: Settings for Q-function plots.
        :param env_plot_settings: Settings for rollout plot.
        """
        Serializable.quick_init(self, locals())
        super(SoftQLearning, self).__init__(**base_kwargs)

        self._env = env

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._alpha = alpha
        self._alpha_scheduler = alpha_scheduler

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
        self._create_policy(policy_class, policy_kwargs)
        self._create_qf(qf_class, qf_kwargs)
        self._create_kernel(kernel_class)

        self._policy_params = self._training_policy.get_params_internal()
        self._qf_params = self._qf_eval.get_params_internal()
        self._target_qf_params = self._qf_td_target.get_params_internal()

        self._training_ops = []
        self._target_ops = []
        self._init_svgd_update()
        self._init_td_update()
        self._init_target_ops()

        self._q_plot_settings = q_plot_settings
        self._env_plot_settings = env_plot_settings
        self._init_figures()

        self._eval_policy = self._training_policy
        self._eval_n_episodes = eval_n_episodes
        self._eval_render = eval_render

        self._sess.run(tf.global_variables_initializer())

    @property
    def policy(self):
        return self._training_policy

    @property
    def env(self):
        return self._env

    @overrides
    def train(self):
        """ Starts the Soft Q-Learning algorithm. """
        self._train(self._env, self._training_policy)

    def _create_placeholders(self):
        """ Creates all necessary placeholders. """
        # We use tf_proxy for the observation placeholder to make it
        # serializable. This is needed to make also the policy serializable.
        self._obs_pl = tp.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._obs_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='next_observation',
        )
        self._actions_pl = tp.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._actions_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='next_actions',
        )

        self._rewards_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='rewards',
        )

        self._terminals_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminals',
        )

        self._alpha_pl = tf.placeholder(
            tf.float32,
            shape=(),
            name='alpha'
        )

    def _create_policy(self, policy_class, policy_kwargs):
        """
        Creates two policies: one for the TD update and one for the SVGD update.
        They share the same parameters, but have different input/output
        dimensions.
        """

        with tf.variable_scope('policy') as scope:
            self._policy_out = policy_class(
                inputs=(self._obs_pl,),
                K=self._kernel_K,
                **policy_kwargs
            )  # N x K x Da

            self._actions_fixed, self._actions_updated = tf.split(
                self._policy_out,
                [self._kernel_K_fixed, self._kernel_K_updated],
                axis=1
            )  # N x (K_fix / K_upd) x Da

            # The gradients should not be back-propagated into the inner
            # empirical expectation.
            self._actions_fixed = tf.stop_gradient(self._actions_fixed)

            scope.reuse_variables()

            # A policy network for rollouts and visualization of action samples.
            training_policy_out = policy_class(
                inputs=(self._obs_pl,),
                K=1,
                **policy_kwargs
            )  # N x Da

            self._training_policy = NNPolicy(
                self._env.spec, self._obs_pl,
                training_policy_out,
            )

            self._visualization_policy = NNPolicy(
                self._env.spec, self._obs_pl,
                self._policy_out,
            )

    def _create_qf(self, qf_class, qf_kwargs):
        """
        Creates three Q-functions: one for the TD update, one for SVGD,
        and one for visualization. They all share the same parameters, but have
        different input/output dimensions. Additionally, the method creates a
        separate network (not sharing weights) that serves as a target network
        for the TD updates.
        """

        with tf.variable_scope('qf') as scope:
            # Actions are normalized, and should reside between -1 and 1. The
            # environment will clip the actions, so we'll encode that as a prior
            # directly into the Q-function.
            clipped_actions = tp.clip_by_value(self._actions_pl, -1, 1)
            self._qf = qf_class(
                inputs=(self._obs_pl, clipped_actions),
                **qf_kwargs
            )  # N x 1

            scope.reuse_variables()

            # SVGD target Q-function. Expand the dimensions to make use of
            # broadcasting (see documentation for NeuralNetwork). This will
            # evaluate the Q-function for each state-action input pair.
            obs_expanded = tf.expand_dims(self._obs_pl, axis=1)  # N x 1 x Do
            qf_unbounded_net = qf_class(
                inputs=(obs_expanded, self._actions_fixed),
                **qf_kwargs
            )  # N x K_fix x 1

            # InputBounds modifies the gradient outside the action boundaries to
            # point back into the action domain. This is needed since SVGD
            # assumes unconstrained target domain, so actions may "leave" their
            # domain temporarily, but the modified gradient will eventually
            # bring them back.
            self._qf_svgd_target = InputBounds(self._actions_fixed,
                                               qf_unbounded_net)

            # Q function for evaluation purposes.
            obs_expanded = tf.expand_dims(self._obs_pl, axis=1)
            actions_expanded = tf.expand_dims(self._actions_pl, axis=0)
            qf_eval_net = qf_class(
                inputs=(obs_expanded, actions_expanded),
                **qf_kwargs
            )

            self._qf_eval = NNQFunction(
                obs_pl=self._obs_pl,
                actions_pl=self._actions_pl,
                q_value=qf_eval_net
            )

        with tf.variable_scope('qf_td_target'):
            # Creates TD target network. Value of the next state is approximated
            # with uniform samples.
            obs_next_expanded = tf.expand_dims(self._obs_next_pl, axis=1)
            # N x 1 x Do
            target_actions = tf.random_uniform((1, self._qf_target_K, self._Da),
                                               -1, 1)  # 1 x K x Da
            self._q_value_td_target = qf_class(
                inputs=(obs_next_expanded, target_actions),
                **qf_kwargs
            )  # N x 1 x 1

            self._qf_td_target = NNQFunction(
                obs_pl=self._obs_next_pl,
                actions_pl=target_actions,
                q_value=self._q_value_td_target
            )

    def _create_kernel(self, kernel_class):
        self._kernel = kernel_class(
            xs=self._actions_fixed,
            ys=self._actions_updated
        )  # N x K_fix x K_upd

    def _init_svgd_update(self):
        """ Creates a TF operation for the SVGD update. """

        # Target log-density.
        log_p = self._qf_svgd_target  # N x K_fix x 1

        grad_log_p = tf.gradients(log_p, self._actions_fixed)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)  # N x K_fix x 1 x Da
        grad_log_p = tf.stop_gradient(grad_log_p)

        kappa = tf.expand_dims(
            self._kernel,
            dim=3
        )  # N x K_fix x K_upd x 1

        kappa_grads = self._kernel.grad  # N x K_fix x K_upd x Da

        # Stein Variational Gradient!
        action_grads = tf.reduce_mean(
            kappa * grad_log_p  # N x K_fix x K_upd x Da
            + self._alpha_pl * kappa_grads,
            reduction_indices=1
        )  # N x K_upd x Da

        # Propagate the gradient through the policy network.
        param_grads = tf.gradients(
            self._actions_updated,
            self._policy_params,
            grad_ys=action_grads
        )

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

        q_curr = self._qf  # N x 1
        q_curr = tf.squeeze(q_curr)  # N
        q_next = self._q_value_td_target  # N x K_td x 1
        n_target_particles = tf.cast(tf.shape(q_next)[1], tf.float32)
        # = self._qf_target_K

        v_next = tf.squeeze(tf.reduce_logsumexp(q_next, axis=1))  # N

        # Importance weights add just a constant to the value, which is
        # irrelevant in terms of the actual policy.
        v_next -= tf.log(n_target_particles)
        v_next += self._Da * np.log(2)

        ys = (tf.squeeze(self._rewards_pl) +
              (1 - self._terminals_pl) * self._discount * v_next)  # N
        ys = tf.stop_gradient(ys)

        td_loss = tf.reduce_mean(tf.square(ys - q_curr))  # 1

        td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=td_loss,
            var_list=self._qf_params
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
    def _init_training(self, env, policy):
        super(SoftQLearning, self)._init_training(env, policy)
        self._qf_td_target.set_param_values(self._qf_eval.get_param_values())

    @overrides
    def _get_training_ops(self, itr):
        ops = self._training_ops

        return ops

    # It is significantly faster to run the training ops and the target update
    # ops in separate sess.run calls compared to running them in a single call.
    # Reason unknown.
    def _get_target_ops(self, itr):
        if itr % self._qf_target_update_interval == 0:
            ops = self._target_ops
        else:
            ops = list()

        return ops

    @overrides
    def _update_feed_dict(
            self, itr, rewards, terminals, obs, actions, next_obs
    ):
        if self._alpha_scheduler:
            alpha = self._alpha_scheduler.apply(self._alpha, itr)
        else:
            alpha = self._alpha

        feeds = {
            self._obs_pl: obs,
            self._actions_pl: actions,
            self._obs_next_pl: next_obs,
            self._rewards_pl: rewards,
            self._terminals_pl: terminals,
            self._alpha_pl: alpha
        }

        return feeds

    def _init_figures(self):
        # Init an environment figure.
        if self._env_plot_settings is not None:
            if "figsize" not in self._env_plot_settings.keys():
                figsize = (7, 7)
            else:
                figsize = self._env_plot_settings['figsize']
            self._fig_env = plt.figure(
                figsize=figsize
            )
            self._ax_env = self._fig_env.add_subplot(111)
            if hasattr(self._env, 'init_plot'):
                self._env.init_plot(self._ax_env)

            # A list for holding line objects created by the environment.
            self._env_lines = []
            self._ax_env.set_xlim(self._env_plot_settings['xlim'])
            self._ax_env.set_ylim(self._env_plot_settings['ylim'])

            # Labelling.
            if "title" in self._env_plot_settings:
                self._ax_env.set_title(self._env_plot_settings["title"])
            if "xlabel" in self._env_plot_settings:
                self._ax_env.set_xlabel(self._env_plot_settings["xlabel"])
            if "ylabel" in self._env_plot_settings:
                self._ax_env.set_ylabel(self._env_plot_settings["ylabel"])

        # Init a figure for the Q-function and action samples.
        if self._q_plot_settings is not None:
            self._q_plot_settings['obs_lst'] = (
                np.array(self._q_plot_settings['obs_lst'])
            )
            n_states = len(self._q_plot_settings['obs_lst'])

            x_size = 5 * n_states
            y_size = 5

            self._fig_q = plt.figure(figsize=(x_size, y_size))

            self._ax_q_lst = []
            for i in range(n_states):
                ax = self._fig_q.add_subplot(100 + n_states * 10 + i + 1)
                ax.set_xlim(self._q_plot_settings['xlim'])
                ax.set_ylim(self._q_plot_settings['ylim'])
                self._ax_q_lst.append(ax)

    @overrides
    def _evaluate(self, epoch):

        logger.log("Collecting samples for evaluation")
        snapshot_dir = logger.get_snapshot_dir()

        paths = rollouts(self._env, self._eval_policy,
                         self._max_path_length, self._eval_n_episodes,
                         self._eval_render)

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self._discount)
             for path in paths]
        )

        total_returns = [
            path['rewards'].sum() for path in paths
        ]

        episode_lengths = [
            len(p['rewards']) for p in paths
        ]

        statistics = OrderedDict([
            ('Epoch', epoch),
            ('Alpha', self._alpha),
            ('DiscountedReturnAvg', average_discounted_return),
            ('TotalReturnAvg', np.mean(total_returns)),
            ('TotalReturnMin', np.min(total_returns)),
            ('TotalReturnMax', np.max(total_returns)),
            ('TotalReturnStd', np.std(total_returns)),
            ('EpisodeLengthAvg', np.mean(episode_lengths)),
            ('EpisodeLengthMin', np.min(episode_lengths)),
            ('EpisodeLengthMax', np.max(episode_lengths)),
            ('EpisodeLengthStd', np.std(episode_lengths))
        ])

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self._env.log_diagnostics(paths)

        # Plot test paths.
        if (hasattr(self._env, 'plot_paths')
                and self._env_plot_settings is not None):
            # Remove previous paths.
            if self._env_lines is not None:
                [path.remove() for path in self._env_lines]
            self._env_lines = self._env.plot_paths(paths, self._ax_env)
            plt.pause(0.001)
            plt.draw()
            if snapshot_dir is not None:
                img_file = os.path.join(snapshot_dir,
                                        'env_itr_%05d.png' % epoch)
                self._fig_env.savefig(img_file, dpi=100)

        # Plot the Q-function level curves and action samples.
        if (hasattr(self._qf_eval, 'plot_level_curves')
                and self._q_plot_settings is not None):
            [ax.clear() for ax in self._ax_q_lst]
            self._qf_eval.plot_level_curves(
                ax_lst=self._ax_q_lst,
                observations=self._q_plot_settings['obs_lst'],
                action_dims=self._q_plot_settings['action_dims'],
                xlim=self._q_plot_settings['xlim'],
                ylim=self._q_plot_settings['ylim'],
            )
            self._visualization_policy.plot_samples(
                self._ax_q_lst, self._q_plot_settings['obs_lst']
            )
            for ax in self._ax_q_lst:
                ax.set_xlim(self._q_plot_settings['xlim'])
                ax.set_ylim(self._q_plot_settings['ylim'])
            plt.pause(0.001)
            plt.draw()
            if snapshot_dir is not None:
                img_file = os.path.join(snapshot_dir,
                                        'q_itr_%05d.png' % epoch)
                self._fig_q.savefig(img_file, dpi=100)

        gc.collect()

    @overrides
    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            policy=self._training_policy,
            qf=self._qf,
            env=self._env,
        )

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d.update({
            "policy_params": self._training_policy.get_param_values(),
            "qf_params": self._qf_eval.get_param_values()
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._qf_eval.set_param_values(d["qf_params"])
        self._training_policy.set_param_values(d["policy_params"])
