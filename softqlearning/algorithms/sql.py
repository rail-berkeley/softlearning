import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel

from .rl_algorithm import RLAlgorithm

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


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
            policy_lr=1E-3,
            qf_lr=1E-3,
            value_n_particles=16,
            td_target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            reward_scale=1,
            save_full_state=False,
    ):
        """
        :param base_kwargs: Keyword arguments for OnlineAlgorithm.
        :param env: Environment object.
        :param value_n_particles: Number of uniform samples used to estimate
            the soft target value for TD learning.
        :param td_target_update_interval: How often (after how many iterations)
            the target network is updated to match the current Q-function.
        :param qf_lr: TD learning rate.
        :param policy_lr: SVGD learning rate.
        :param kernel_n_particles: Total number of particles per state used in
            the SVGD updates.
        :param kernel_update_ratio: The ratio of SVGD particles used for the
            computation of the inner/outer empirical expectation.
        :param discount: Discount factor.
        """
        Serializable.quick_init(self, locals())
        super().__init__(**base_kwargs)

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

        self._observation_dim = self.env.observation_space.flat_dim
        self._action_dim = self.env.action_space.flat_dim

        self._create_placeholders()

        self._training_ops = []
        self._target_ops = []

        self._create_td_update()
        self._create_svgd_update()
        self._create_target_ops()

        self._sess.run(tf.global_variables_initializer())

    @overrides
    def train(self):
        """ Starts the Soft Q-Learning algorithm. """
        self._train(self.env, self.policy, self.pool)

    def _create_placeholders(self):
        """ Creates all necessary placeholders. """

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
        """ Creates a TF operation for the TD update. """

        with tf.variable_scope('target'):
            """
            Create TD target by approximating the next value with uniform
            samples.
            """
            target_actions = tf.random_uniform(
                (1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(
                observations=self._next_observations_ph[:, None, :],
                actions=target_actions)
            assert_shape(q_value_targets, [None, self._value_n_particles])

        self._q_values = self.qf.output_for(
            self._observations_ph, self._actions_pl, reuse=True)
        assert_shape(self._q_values, [None])

        # Eq. (10):
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        assert_shape(next_value, [None])

        # Importance weights add just a constant to the value, which is
        # irrelevant in terms of the actual policy.
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_dim * np.log(2)

        # Qhat_soft in Eq. (11):
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl + (
            1 - self._terminals_pl) * self._discount * next_value)
        assert_shape(ys, [None])

        # Eq (11):
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)

        td_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=bellman_residual, var_list=self.qf.get_params_internal())

        self._training_ops.append(td_train_op)
        self._bellman_residual = bellman_residual

    def _create_svgd_update(self):
        """ Creates a TF operation for the SVGD update. """

        actions = self.policy.actions_for(
            observations=self._observations_ph,
            n_action_samples=self._kernel_n_particles)
        assert_shape(actions,
                     [None, self._kernel_n_particles, self._action_dim])

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

        # Target log-density. Q_soft in eq. (13):
        squash_correction = tf.reduce_sum(
            tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)
        assert_shape(grad_log_p, [None, n_fixed_actions, 1, self._action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in eq. (13).
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient! Eq. (13):
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, self._action_dim])

        # Propagate the gradient through the policy network. Eq. (14):
        gradients = tf.gradients(
            updated_actions,
            self.policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)
        ])

        optimizer = tf.train.AdamOptimizer(self._policy_lr)
        svgd_training_op = optimizer.minimize(
            loss=-surrogate_loss, var_list=self.policy.get_params_internal())

        self._training_ops.append(svgd_training_op)

    def _create_target_ops(self):
        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')

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
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

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
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
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
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            return {'epoch': epoch, 'algo': self}

        return {
            'epoch': epoch,
            'policy': self.policy,
            'qf': self.qf,
            'env': self.env,
        }

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        state = Serializable.__getstate__(self)
        state.update({
            'qf-params': self.qf.get_param_values(),
            'policy-params': self.policy.get_param_values(),
            'pool': self.pool.__getstate__(),
            'env': self.env.__getstate__(),
        })
        return state

    def __setstate__(self, state):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, state)
        self.qf.set_param_values(state['qf-params'])
        self.policy.set_param_values(state['policy-params'])
        self.pool.__setstate__(state['pool'])
        self.env.__setstate__(state['env'])
