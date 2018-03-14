"""Diversity Is All You Need (DIAYN)"""

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from sac.algos.sac import SAC
from sac.misc import tf_utils, utils
from sac.misc.sampler import rollouts
from sac.policies.hierarchical_policy import FixedOptionPolicy

from collections import deque
import gtimer as gt
import json
import numpy as np
import os
import scipy.stats
import tensorflow as tf


EPS = 1E-6

class DIAYN(SAC):

    def __init__(self,
                 base_kwargs,
                 env,
                 policy,
                 discriminator,
                 qf,
                 vf,
                 pool,
                 plotter=None,
                 lr=3E-3,
                 scale_entropy=1,
                 discount=0.99,
                 tau=0.01,
                 num_skills=20,
                 save_full_state=False,
                 find_best_skill_interval=10,
                 best_skill_n_rollouts=10,
                 learn_p_z=False,
                 include_actions=False,
                 add_p_z=True):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            discriminator: (`rllab.NNPolicy`): A discriminator for z.
            qf (`ValueFunction`): Q-function approximator.
            vf (`ValueFunction`): Soft value function approximator.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            scale_entropy (`float`): Scaling factor for entropy.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            num_skills (`int`): Number of skills/options to learn.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
            find_best_skill_interval (`int`): How often to recompute the best
                skill.
            best_skill_n_rollouts (`int`): When finding the best skill, how
                many rollouts to do per skill.
            include_actions (`bool`): Whether to pass actions to the
                discriminator.
            add_p_z (`bool`): Whether th include log p(z) in the pseudo-reward.
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._discriminator = discriminator
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._discriminator_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_entropy = scale_entropy
        self._discount = discount
        self._tau = tau
        self._num_skills = num_skills
        self._p_z = np.full(num_skills, 1.0 / num_skills)
        self._find_best_skill_interval = find_best_skill_interval
        self._best_skill_n_rollouts = best_skill_n_rollouts
        self._learn_p_z = learn_p_z
        self._save_full_state = save_full_state
        self._include_actions = include_actions
        self._add_p_z = add_p_z

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_discriminator_update()
        self._init_target_ops()


        self._sess.run(tf.global_variables_initializer())

    def _init_placeholders(self):
        """Create input placeholders for the DIAYN algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - terminals
            - zs
        """

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do + self._num_skills],
            name='observation',
        )

        self._obs_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do + self._num_skills],
            name='next_observation',
        )
        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._terminal_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminals',
        )

        self._p_z_pl = tf.placeholder(
            tf.float32,
            shape=[self._num_skills],
            name='p_z',
        )

    def _sample_z(self):
        """Samples z from p(z), using probabilities in self._p_z."""
        return np.random.choice(self._num_skills, p=self._p_z)

    def _split_obs(self):
        return tf.split(self._obs_pl, [self._Do, self._num_skills], 1)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        self._qf_t = self._qf.get_output_for(
            self._obs_pl, self._action_pl, reuse=True)  # N

        (obs, z_one_hot) = self._split_obs()
        if self._include_actions:
            logits = self._discriminator.get_output_for(obs, self._action_pl,
                                                        reuse=True)
        else:
            logits = self._discriminator.get_output_for(obs, reuse=True)
        reward_pl = -1 * tf.nn.softmax_cross_entropy_with_logits(labels=z_one_hot,
                                                                 logits=logits)
        reward_pl = tf.check_numerics(reward_pl, 'Check numerics (1): reward_pl')
        p_z = tf.reduce_sum(self._p_z_pl * z_one_hot, axis=1)
        log_p_z = tf.log(p_z + EPS)
        self._log_p_z = log_p_z
        if self._add_p_z:
            reward_pl -= log_p_z
            reward_pl = tf.check_numerics(reward_pl, 'Check numerics: reward_pl')
        self._reward_pl = reward_pl

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._obs_next_pl)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            reward_pl + (1 - self._terminal_pl) * self._discount * vf_next_target_t
        )  # N

        self._td_loss_t = 0.5 * tf.reduce_mean((ys - self._qf_t)**2)

        qf_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss_t,
            var_list=self._qf.get_params_internal()
        )

        self._training_ops.append(qf_train_op)

    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """

        self._policy_dist = self._policy.get_distribution_for(
            self._obs_pl, reuse=True)
        log_pi_t = self._policy_dist.log_p_t  # N

        self._vf_t = self._vf.get_output_for(self._obs_pl, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        log_target_t = self._qf.get_output_for(
            self._obs_pl, tf.tanh(self._policy_dist.x_t), reuse=True)  # N
        corr = self._squash_correction(self._policy_dist.x_t)
        corr = tf.check_numerics(corr, 'Check numerics: corr')

        scaled_log_pi = self._scale_entropy * (log_pi_t - corr)

        self._kl_surrogate_loss_t = tf.reduce_mean(log_pi_t * tf.stop_gradient(
            scaled_log_pi - log_target_t + self._vf_t)
        )

        self._vf_loss_t = 0.5 * tf.reduce_mean(
            (self._vf_t - tf.stop_gradient(log_target_t - scaled_log_pi))**2
        )

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=self._kl_surrogate_loss_t + self._policy_dist.reg_loss_t,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)


    def _init_discriminator_update(self):
        (obs, z_one_hot) = self._split_obs()
        if self._include_actions:
            logits = self._discriminator.get_output_for(obs, self._action_pl,
                                                        reuse=True)
        else:
            logits = self._discriminator.get_output_for(obs, reuse=True)

        self._discriminator_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=z_one_hot,
                                                    logits=logits)
        )
        optimizer = tf.train.AdamOptimizer(self._discriminator_lr)
        discriminator_train_op = optimizer.minimize(
            loss=self._discriminator_loss,
            var_list=self._discriminator.get_params_internal()
        )
        self._training_ops.append(discriminator_train_op)


    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._obs_pl: batch['observations'],
            self._action_pl: batch['actions'],
            self._obs_next_pl: batch['next_observations'],
            self._terminal_pl: batch['terminals'],
            self._p_z_pl: self._p_z,
        }

        return feed_dict

    def _get_best_single_option_policy(self):
        best_returns = float('-inf')
        best_z = None
        for z in range(self._num_skills):
            fixed_z_policy = FixedOptionPolicy(self._policy, self._num_skills, z)
            paths = rollouts(self._eval_env, fixed_z_policy,
                             self._max_path_length, self._best_skill_n_rollouts,
                             render=False)
            total_returns = np.mean([path['rewards'].sum() for path in paths])
            if total_returns > best_returns:
                best_returns = total_returns
                best_z = z
        return FixedOptionPolicy(self._policy, self._num_skills, best_z)

    def _save_traces(self, filename):
        utils._make_dir(filename)
        obs_vec = []
        for z in range(self._num_skills):
            fixed_z_policy = FixedOptionPolicy(self._policy,
                                               self._num_skills, z)
            paths = rollouts(self._eval_env, fixed_z_policy,
                             self._max_path_length, n_paths=3,
                             render=False)
            obs_vec.append([path['observations'].tolist() for path in paths])

        with open(filename, 'w') as f:
            json.dump(obs_vec, f)


    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        We always use the most recent policy, but for computational efficiency
        we sometimes use a stale version of the metapolicy.
        During evaluation, our policy expects an un-augmented observation.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        if epoch % self._find_best_skill_interval == 0:
            self._single_option_policy = self._get_best_single_option_policy()
        for (policy, policy_name) in [(self._single_option_policy, 'best_single_option_policy')]:
            with logger.tabular_prefix(policy_name + '/'), logger.prefix(policy_name + '/'):
                with self._policy.deterministic(self._eval_deterministic):
                    if self._eval_render:
                        paths = rollouts(self._eval_env, policy,
                                         self._max_path_length, self._eval_n_episodes,
                                         render=True, render_mode='rgb_array')
                    else:
                        paths = rollouts(self._eval_env, policy,
                                         self._max_path_length, self._eval_n_episodes)

                total_returns = [path['rewards'].sum() for path in paths]
                episode_lengths = [len(p['rewards']) for p in paths]

                logger.record_tabular('return-average', np.mean(total_returns))
                logger.record_tabular('return-min', np.min(total_returns))
                logger.record_tabular('return-max', np.max(total_returns))
                logger.record_tabular('return-std', np.std(total_returns))
                logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
                logger.record_tabular('episode-length-min', np.min(episode_lengths))
                logger.record_tabular('episode-length-max', np.max(episode_lengths))
                logger.record_tabular('episode-length-std', np.std(episode_lengths))

                self._eval_env.log_diagnostics(paths)

        batch = self._pool.random_batch(self._batch_size)
        self.log_diagnostics(batch)

    def _train(self, env, policy, pool):
        """When training our policy expects an augmented observation."""
        self._init_training(env, policy, pool)

        with self._sess.as_default():
            observation = env.reset()
            policy.reset()
            log_p_z_episode = []  # Store log_p_z for this episode
            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0

            if self._learn_p_z:
                log_p_z_list = [deque(maxlen=self._max_path_length) for _ in range(self._num_skills)]

            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)


                path_length_list = []
                z = self._sample_z()
                aug_obs = utils.concat_obs_z(observation, z, self._num_skills)

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    action, _ = policy.get_action(aug_obs)

                    if self._learn_p_z:
                        (obs, _) = utils.split_aug_obs(aug_obs, self._num_skills)
                        feed_dict = {self._discriminator._obs_pl: obs[None],
                                     self._discriminator._action_pl: action[None]}
                        logits = tf_utils.get_default_session().run(
                            self._discriminator._output_t, feed_dict)[0]
                        log_p_z = np.log(utils._softmax(logits)[z])
                        if self._learn_p_z:
                            log_p_z_list[z].append(log_p_z)

                    next_ob, reward, terminal, info = env.step(action)
                    aug_next_ob = utils.concat_obs_z(next_ob, z,
                                                     self._num_skills)
                    path_length += 1
                    path_return += reward

                    self._pool.add_sample(
                        aug_obs,
                        action,
                        reward,
                        terminal,
                        aug_next_ob,
                    )

                    if terminal or path_length >= self._max_path_length:
                        path_length_list.append(path_length)
                        observation = env.reset()
                        policy.reset()
                        log_p_z_episode = []
                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return

                        path_return = 0
                        n_episodes += 1


                    else:
                        aug_obs = aug_next_ob
                    gt.stamp('sample')

                    if self._pool.size >= self._min_pool_size:
                        for i in range(self._n_train_repeat):
                            batch = self._pool.random_batch(self._batch_size)
                            self._do_training(iteration, batch)

                    gt.stamp('train')

                if self._learn_p_z:
                    print('learning p(z)')
                    for z in range(self._num_skills):
                        if log_p_z_list[z]:
                            print('\t skill = %d, min=%.2f, max=%.2f, mean=%.2f, len=%d' % (z, np.min(log_p_z_list[z]), np.max(log_p_z_list[z]), np.mean(log_p_z_list[z]), len(log_p_z_list[z])))
                    log_p_z = [np.mean(log_p_z) if log_p_z else np.log(1.0 / self._num_skills) for log_p_z in log_p_z_list]
                    print('log_p_z: %s' % log_p_z)
                    self._p_z = utils._softmax(log_p_z)


                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('episodes', n_episodes)
                logger.record_tabular('max-path-return', max_path_return)
                logger.record_tabular('last-path-return', last_path_return)
                logger.record_tabular('pool-size', self._pool.size)
                logger.record_tabular('path-length', np.mean(path_length_list))

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            env.terminate()


    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, the TD-loss (mean squared Bellman error), and the
        discriminator loss (cross entropy) for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(batch)
        log_pairs = [
            ('qf', self._qf_t),
            ('vf', self._vf_t),
            ('bellman-error', self._td_loss_t),
            ('discriminator-loss', self._discriminator_loss),
            ('vf-loss', self._vf_loss_t),
            ('kl-surrogate-loss', self._kl_surrogate_loss_t),
            ('policy-reg-loss', self._policy_dist.reg_loss_t),
            ('discriminator_reward', self._reward_pl),
            ('log_p_z', self._log_p_z),
        ]
        log_ops = [op for (name, op) in log_pairs]
        log_names = [name for (name, op) in log_pairs]
        log_vals = self._sess.run(log_ops, feed_dict)
        for (name, val) in zip(log_names, log_vals):
            if np.isscalar(val):
                logger.record_tabular(name, val)
            else:
                logger.record_tabular('%s-avg' % name, np.mean(val))
                logger.record_tabular('%s-min' % name, np.min(val))
                logger.record_tabular('%s-max' % name, np.max(val))
                logger.record_tabular('%s-std' % name, np.std(val))
        logger.record_tabular('z-entropy', scipy.stats.entropy(self._p_z))

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
            return dict(
                epoch=epoch,
                algo=self
            )
        else:
            return dict(
                epoch=epoch,
                policy=self._policy,
                qf=self._qf,
                vf=self._vf,
                env=self._env,
                discriminator=self._discriminator,
            )

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf-params': self._qf.get_param_values(),
            'vf-params': self._vf.get_param_values(),
            'discriminator-params': self._discriminator.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf.set_param_values(d['qf-params'])
        self._vf.set_param_values(d['qf-params'])
        self._discriminator.set_param_values(d['discriminator-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
