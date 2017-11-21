import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from sac.algos.base import RLAlgorithm


class SAC(RLAlgorithm, Serializable):
    """
    """
    def __init__(
            self,
            base_kwargs,

            env,
            policy,
            qf,
            vf,
            pool,
            plotter=None,

            policy_lr=1E-3,
            qf_lr=1E-3,
            vf_lr=1E-3,

            discount=0.99,
            qf_target_update_interval=1,
            tau=0,

            save_full_state=False,
    ):
        """
        """
        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._vf_lr = vf_lr

        self._discount = discount
        self._qf_target_update_interval = qf_target_update_interval
        self._tau = tau

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        self._sess.run(tf.global_variables_initializer())

    @overrides
    def train(self):
        self._train(self._env, self._policy, self._pool)

    def _init_placeholders(self):
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
        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
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

    def _init_critic_update(self):
        """
        """
        self._qf_t = self._qf.get_output_for(
            self._obs_pl, self._action_pl, reuse=True)  # N

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._obs_next_pl)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            self._reward_pl +
            (1 - self._terminal_pl) * self._discount * vf_next_target_t
        )  # N

        self._td_loss_t = 0.5 * tf.reduce_mean((ys - self._qf_t)**2)

        self._training_ops.append(tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss_t,
            var_list=self._qf.get_params_internal()
        ))

    def _init_actor_update(self):
        """
        """
        policy_dist = self._policy.get_distribution_for(
            self._obs_pl, reuse=True)
        log_pi_t = policy_dist.log_p_t  # N

        self._vf_t = self._vf.get_output_for(self._obs_pl, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        log_target_t = self._qf.get_output_for(
            self._obs_pl, tf.tanh(policy_dist.x_t), reuse=True)  # N
        corr = self._squash_correction(policy_dist.x_t)

        kl_loss_t = tf.reduce_mean(log_pi_t * tf.stop_gradient(
            log_pi_t - log_target_t - corr + self._vf_t))

        self._vf_loss_t = 0.5 * tf.reduce_mean(
            (self._vf_t - tf.stop_gradient(log_target_t - log_pi_t + corr))**2)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=kl_loss_t + policy_dist.reg_loss_t,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)

    @staticmethod
    def _squash_correction(t):
        EPS = 1E-6
        return tf.reduce_sum(tf.log(1 - tf.tanh(t) ** 2 + EPS), axis=1)

    def _init_target_ops(self):
        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(tgt, self._tau * tgt + (1 - self._tau) * src)
            for tgt, src in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super(SAC, self)._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, itr, batch):
        feeds = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feeds)
        if itr % self._qf_target_update_interval == 0:
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        feeds = {
            self._obs_pl: batch['observations'],
            self._action_pl: batch['actions'],
            self._obs_next_pl: batch['next_observations'],
            self._reward_pl: batch['rewards'],
            self._terminal_pl: batch['terminals'],
        }

        return feeds

    @overrides
    def log_diagnostics(self, batch):
        feeds = self._get_feed_dict(batch)
        qf, vf, td_loss = self._sess.run(
            [self._qf_t, self._vf_t, self._td_loss_t], feeds)

        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('vf-avg', np.mean(vf))
        logger.record_tabular('vf-std', np.std(vf))
        logger.record_tabular('mean-sq-bellman-error', td_loss)

        self._policy.log_diagnostics(batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
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
            )

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d.update({
            'qf-params': self._qf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._qf.set_param_values(d['qf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
