import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm

EPS = 1E-6


class SAC(RLAlgorithm, Serializable):
    """Soft Actor-Critic (SAC)

    Example:
    ```python

    env = normalize(SwimmerEnv())

    pool = SimpleReplayPool(env_spec=env.spec, max_pool_size=1E6)

    base_kwargs = dict(
        min_pool_size=1000,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=1000,
        batch_size=64,
        scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = 100
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    policy = GMMPolicy(
        env_spec=env.spec,
        K=2,
        hidden_layers=(M, M),
        qf=qf,
        reg=0.001
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3E-4,
        discount=0.99,
        tau=0.01,

        save_full_state=False
    )

    algorithm.train()
    ```

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," Deep Learning Symposium, NIPS 2017.
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

            lr=3E-3,
            scale_reward=1,
            discount=0.99,
            tau=0.01,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            qf (`ValueFunction`): Q-function approximator.
            vf (`ValueFunction`): Soft value function approximator.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            scale_reward (`float`): Scaling factor for raw reward.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.

            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._tau = tau

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        """Initiate training of the SAC instance."""

        self._train(self._env, self._policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """

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
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        self._qf_t = self._qf.get_output_for(
            self._obs_pl, self._action_pl, reuse=True)  # N

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._obs_next_pl)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            self._scale_reward * self._reward_pl +
            (1 - self._terminal_pl) * self._discount * vf_next_target_t
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

        policy_dist = self._policy.get_distribution_for(
            self._obs_pl, reuse=True)
        log_pi_t = policy_dist.log_p_t  # N

        self._vf_t = self._vf.get_output_for(self._obs_pl, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        log_target_t = self._qf.get_output_for(
            self._obs_pl, tf.tanh(policy_dist.x_t), reuse=True)  # N
        corr = self._squash_correction(policy_dist.x_t)

        kl_surrogate_loss_t = tf.reduce_mean(log_pi_t * tf.stop_gradient(
            log_pi_t - log_target_t - corr + self._vf_t))

        self._vf_loss_t = 0.5 * tf.reduce_mean(
            (self._vf_t - tf.stop_gradient(log_target_t - log_pi_t + corr))**2)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=kl_surrogate_loss_t + policy_dist.reg_loss_t,
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
        return tf.reduce_sum(tf.log(1 - tf.tanh(t) ** 2 + EPS), axis=1)

    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super(SAC, self)._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, itr, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)
        self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._obs_pl: batch['observations'],
            self._action_pl: batch['actions'],
            self._obs_next_pl: batch['next_observations'],
            self._reward_pl: batch['rewards'],
            self._terminal_pl: batch['terminals'],
        }

        return feed_dict

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(batch)
        qf, vf, td_loss = self._sess.run(
            [self._qf_t, self._vf_t, self._td_loss_t], feed_dict)

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
            )

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf-params': self._qf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf.set_param_values(d['qf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
