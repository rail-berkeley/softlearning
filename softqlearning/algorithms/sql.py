import numpy as np
import tensorflow as tf

from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.misc import tf_utils

from .rl_algorithm import RLAlgorithm

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class SQL(RLAlgorithm):
    """Soft Q-learning (SQL).

    Example:
        See `examples/mujoco_all_sql.py`.

    Reference:
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """

    def __init__(
            self,
            base_kwargs,
            env,
            pool,
            qf,
            policy,
            plotter=None,
            policy_lr=1E-3,
            qf_lr=1E-3,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=1,
            use_saved_qf=False,
            use_saved_policy=False,
            save_full_state=False,
            train_qf=True,
            train_policy=True,
    ):
        """
        Args:
            base_kwargs (dict): Dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`rllab.Env`): rllab environment object.
            pool (`PoolBase`): Replay buffer to add gathered samples to.
            qf (`NNQFunction`): Q-function approximator.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            qf_lr (`float`): Learning rate used for the Q-function approximator.
            value_n_particles (`int`): The number of action samples used for
                estimating the value of next state.
            td_target_update_interval (`int`): How often the target network is
                updated to match the current Q-function.
            kernel_fn (function object): A function object that represents
                a kernel function.
            kernel_n_particles (`int`): Total number of particles per state
                used in SVGD updates.
            kernel_update_ratio ('float'): The ratio of SVGD particles used for
                the computation of the inner/outer empirical expectation.
            discount ('float'): Discount factor.
            reward_scale ('float'): A factor that scales the raw rewards.
                Useful for adjusting the temperature of the optimal Boltzmann
                distribution.
            use_saved_qf ('boolean'): If true, use the initial parameters provided
                in the Q-function instead of reinitializing.
            use_saved_policy ('boolean'): If true, use the initial parameters provided
                in the policy instead of reinitializing.
            save_full_state ('boolean'): If true, saves the full algorithm
                state, including the replay buffer.
        """
        super(SQL, self).__init__(**base_kwargs)

        self.env = env
        self.pool = pool
        self.qf = qf
        self.policy = policy
        self.plotter = plotter

        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._reward_scale = reward_scale

        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_qf = train_qf
        self._train_policy = train_policy

        self._observation_dim = self.env.observation_space.flat_dim
        self._action_dim = self.env.action_space.flat_dim

        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_td_update()
        self._create_svgd_update()
        self._create_target_ops()

        if use_saved_qf:
            saved_qf_params = qf.get_param_values()
        if use_saved_policy:
            saved_policy_params = policy.get_param_values()

        self._sess = tf_utils.get_default_session()
        self._sess.run(tf.global_variables_initializer())

        if use_saved_qf:
            self.qf.set_param_values(saved_qf_params)
        if use_saved_policy:
            self.policy.set_param_values(saved_policy_params)

    def _create_placeholders(self):
        """Create all necessary placeholders."""

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations')

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='next_observations')

        self._actions_pl = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='actions')

        self._next_actions_ph = tf.placeholder(
            tf.float32, shape=[None, self._action_dim], name='next_actions')

        self._rewards_pl = tf.placeholder(
            tf.float32, shape=[None], name='rewards')

        self._terminals_pl = tf.placeholder(
            tf.float32, shape=[None], name='terminals')

    def _create_td_update(self):
        """Create a minimization operation for Q-function update."""

        with tf.variable_scope('target'):
            # The value of the next state is approximated with uniform samples.
            target_actions = tf.random_uniform(
                (1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])

        self._q_values = self.qf.output_for(
            self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(self._q_values, [None])

        # Equation 10:
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])

        # Importance weights add just a constant to the value.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)

        # \hat Q in Equation 11:
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])

        # Equation 11:
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)

        if self._train_qf:
            td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
                loss=bellman_residual, var_list=self.qf.get_params_internal())
            self._training_ops.append(td_train_op)

        self._bellman_residual = bellman_residual

    def _create_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""

        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._kernel_n_particles,
            reuse=True)
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._action_dim])

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(
            self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions = tf.split(
            actions, [n_fixed_actions, n_updated_actions], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        assert_shape(fixed_actions, [None, n_fixed_actions, self._action_dim])
        assert_shape(updated_actions,
                     [None, n_updated_actions, self._action_dim])

        svgd_target_values = self.qf.output_for(
            self._observations_ph[:, None, :], fixed_actions, reuse=True)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._action_dim])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self.policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)
        ])

        if self._train_policy:
            optimizer = tf.train.AdamOptimizer(self._policy_lr)
            svgd_training_op = optimizer.minimize(
                loss=-surrogate_loss,
                var_list=self.policy.get_params_internal())
            self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        """Create tensorflow operation for updating the target Q-function."""
        if not self._train_qf:
            return

        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')

        self._target_ops = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]

    # TODO: do not pass, policy, and pool to `__init__` directly.
    def train(self):
        self._train(self.env, self.policy, self.pool)

    @overrides
    def _init_training(self):
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._qf_target_update_interval == 0 and self._train_qf:
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        feeds = {
            self._observations_ph: batch['observations'],
            self._actions_pl: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_pl: batch['rewards'],
            self._terminals_pl: batch['terminals'],
        }

        return feeds

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information.

        Records the mean and standard deviation of Q-function and the
        squared Bellman residual of the  s (mean squared Bellman error)
        for a sample batch.

        Also call the `draw` method of the plotter, if plotter is defined.
        """

        feeds = self._get_feed_dict(batch)
        qf, bellman_residual = self._sess.run(
            [self._q_values, self._bellman_residual], feeds)

        logger.record_tabular('qf-avg', np.mean(qf))
        logger.record_tabular('qf-std', np.std(qf))
        logger.record_tabular('mean-sq-bellman-error', bellman_residual)

        self.policy.log_diagnostics(batch)
        if self.plotter:
            self.plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SQL algorithm.

        If `self._save_full_state == True`, returns snapshot including the
        replay buffer. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, and environment instances.
        """

        state = {
            'epoch': epoch,
            'policy': self.policy,
            'qf': self.qf,
            'env': self.env,
        }

        if self._save_full_state:
            state.update({'replay_buffer': self.pool})

        return state
