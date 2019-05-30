from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel

from .rl_algorithm import RLAlgorithm
from .sac import td_target

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class SQL(RLAlgorithm):
    """Soft Q-learning (SQL).

    Example:
        See `examples/development.py`.

    References
    ----------
    [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            policy_lr=3e-4,
            Q_lr=3e-4,
            value_n_particles=16,
            target_update_interval=1,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            discount=0.99,
            tau=5e-3,
            reward_scale=1,
            use_saved_Q=False,
            use_saved_policy=False,
            save_full_state=False,
            train_Q=True,
            train_policy=True,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment object used for training.
            policy: A policy function approximator.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            Q_lr (`float`): Learning rate used for the Q-function approximator.
            value_n_particles (`int`): The number of action samples used for
                estimating the value of next state.
            target_update_interval (`int`): How often the target network is
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
            use_saved_Q ('boolean'): If true, use the initial parameters provided
                in the Q-function instead of reinitializing.
            use_saved_policy ('boolean'): If true, use the initial parameters provided
                in the policy instead of reinitializing.
            save_full_state ('boolean'): If true, saves the full algorithm
                state, including the replay pool.
        """
        super(SQL, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._Q_lr = Q_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._tau = tau
        self._reward_scale = reward_scale

        self._value_n_particles = value_n_particles
        self._Q_target_update_interval = target_update_interval

        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio

        self._save_full_state = save_full_state
        self._train_Q = train_Q
        self._train_policy = train_policy

        if use_saved_Q:
            saved_Q_weights = tuple(Q.get_weights() for Q in self._Qs)
        if use_saved_policy:
            saved_policy_weights = policy.get_weights()

        self._session.run(tf.compat.v1.global_variables_initializer())

        if use_saved_Q:
            for Q, Q_weights in zip(self._Qs, saved_Q_weights):
                Q.set_weights(Q_weights)
        if use_saved_policy:
            self._policy.set_weights(saved_policy_weights)

        self._build()

    def _build(self):
        super(SQL, self)._build()

        self._init_td_update()
        self._init_svgd_update()
        self._init_diagnostics_ops()

    def _get_Q_target(self):
        next_Q_observations = {
            name: tf.reshape(
                tf.tile(
                    self._placeholders['next_observations'][name][:, tf.newaxis, :],
                    (1, self._value_n_particles, 1)),
                (-1, *self._placeholders['next_observations'][name].shape[1:]))
            for name in self._Qs[0].observation_keys
        }

        action_shape = self._placeholders['actions'].shape[1:].as_list()
        target_actions = tf.random.uniform(
            (1, self._value_n_particles, *action_shape), -1, 1)
        target_actions = tf.tile(
            target_actions,
            (tf.shape(self._placeholders['actions'])[0], 1, 1))
        target_actions = tf.reshape(target_actions, (-1, *action_shape))

        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': target_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)

        assert_shape(min_next_Q, (None, 1))

        min_Q_next_target = tf.reshape(min_next_Q, (-1, self._value_n_particles))

        assert_shape(min_Q_next_target, (None, self._value_n_particles))

        # Equation 10 in [1]:
        next_values = tf.reduce_logsumexp(
            min_Q_next_target, keepdims=True, axis=1)

        assert_shape(next_values, [None, 1])

        # Importance weights add just a constant to the value.
        next_values -= tf.math.log(
            tf.cast(self._value_n_particles, tf.float32))
        next_values += np.prod(action_shape) * np.log(2)

        assert_shape(next_values, [None, 1])

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        # \hat Q in Equation 11 in [1]:
        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_td_update(self):
        """Create a minimization operation for Q-function update."""
        Q_target = self._get_Q_target()
        assert_shape(Q_target, [None, 1])

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_actions = self._placeholders['actions']
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': Q_actions})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        for Q_value in self._Q_values:
            assert_shape(Q_value, [None, 1])

        # Equation 11 in [1]:
        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        if self._train_Q:
            self._Q_optimizers = tuple(
                tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self._Q_lr,
                    name='{}_{}_optimizer'.format(Q._name, i)
                ) for i, Q in enumerate(self._Qs))
            Q_training_ops = tuple(
                Q_optimizer.minimize(
                    loss=Q_loss, var_list=Q.trainable_variables)
                for i, (Q, Q_loss, Q_optimizer)
                in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

            self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_svgd_update(self):
        """Create a minimization operation for policy update (SVGD)."""

        policy_inputs = flatten_input_structure({
            name: tf.reshape(
                tf.tile(
                    self._placeholders['observations'][name][:, None, :],
                    (1, self._kernel_n_particles, 1)),
                (-1, *self._placeholders['observations'][name].shape[1:]))
            for name in self._policy.observation_keys
        })
        actions = self._policy.actions(policy_inputs)
        action_shape = actions.shape[1:]
        actions = tf.reshape(
            actions, (-1, self._kernel_n_particles, *action_shape))

        assert_shape(
            actions, (None, self._kernel_n_particles, *action_shape))

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
        assert_shape(fixed_actions,
                     [None, n_fixed_actions, *action_shape])
        assert_shape(updated_actions,
                     [None, n_updated_actions, *action_shape])

        Q_observations = {
            name: tf.reshape(
                    tf.tile(
                        self._placeholders['observations'][name][:, None, :],
                        (1, n_fixed_actions, 1)),
                    (-1, *self._placeholders['observations'][name].shape[1:]))
            for name in self._policy.observation_keys
        }
        Q_actions = tf.reshape(fixed_actions, (-1, *action_shape))
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': Q_actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)
        svgd_target_values = tf.reshape(
            min_Q_log_target, (-1, n_fixed_actions, 1))

        # Target log-density. Q_soft in Equation 13:
        assert self._policy._squash
        squash_correction = tf.reduce_sum(
            tf.math.log(1 - fixed_actions ** 2 + EPS), axis=-1, keepdims=True)
        log_probs = svgd_target_values + squash_correction

        grad_log_probs = tf.gradients(log_probs, fixed_actions)[0]
        grad_log_probs = tf.expand_dims(grad_log_probs, axis=2)
        grad_log_probs = tf.stop_gradient(grad_log_probs)
        assert_shape(grad_log_probs,
                     [None, n_fixed_actions, 1, *action_shape])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = kernel_dict["output"][..., tf.newaxis]
        assert_shape(kappa, [None, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_probs + kernel_dict["gradient"], axis=1)
        assert_shape(action_gradients,
                     [None, n_updated_actions, *action_shape])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self._policy.trainable_variables,
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self._policy.trainable_variables, gradients)
        ])

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name='policy_optimizer'
        )

        if self._train_policy:
            svgd_training_op = self._policy_optimizer.minimize(
                loss=-surrogate_loss,
                var_list=self._policy.trainable_variables)
            self._training_ops.update({
                'svgd': svgd_training_op
            })

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Run the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._Q_target_update_interval == 0 and self._train_Q:
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        evaluation_paths,
                        training_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(flatten_input_structure({
                name: batch['observations'][name]
                for name in self._policy.observation_keys
            })).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        return {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
        }
