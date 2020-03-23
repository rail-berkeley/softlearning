from copy import copy
from collections import OrderedDict

import tensorflow as tf

from softlearning.misc.kernel import adaptive_isotropic_gaussian_kernel

from .rl_algorithm import RLAlgorithm
from .sac import td_targets

EPS = 1e-6


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
        self._Q_targets = tuple(copy(Q) for Q in Qs)
        self._update_target(tau=tf.constant(1.0))

        self._plotter = plotter

        self._policy_lr = policy_lr
        self._Q_lr = Q_lr
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

        if use_saved_Q:
            for Q, Q_weights in zip(self._Qs, saved_Q_weights):
                Q.set_weights(Q_weights)
        if use_saved_policy:
            self._policy.set_weights(saved_policy_weights)

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, next_observations, actions, rewards, terminals):
        expanded_next_observations = type(next_observations)((
            (key, tf.reshape(
                tf.tile(value[:, tf.newaxis, :],
                        (1, self._value_n_particles, 1)),
                (-1, *value.shape[1:])))
            for key, value in next_observations.items()
        ))

        action_shape = tf.shape(actions)[1:]
        target_actions = tf.random.uniform(
            tf.concat(([1, self._value_n_particles], action_shape), axis=0),
            -1.0,
            1.0)
        target_actions = tf.tile(
            target_actions,
            (tf.shape(actions)[0], 1, 1))
        target_actions = tf.reshape(
            target_actions, tf.concat(([-1], action_shape), axis=0))

        next_Qs_values = tuple(
            Q.values(expanded_next_observations, target_actions)
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)

        min_Q_next_target = tf.reshape(min_next_Q, (-1, self._value_n_particles))

        # Equation 10 in [1]:
        next_values = tf.reduce_logsumexp(
            min_Q_next_target, keepdims=True, axis=1)

        # Importance weights add just a constant to the value.
        next_values -= tf.math.log(
            tf.cast(self._value_n_particles, tf.float32))
        next_values += tf.math.log(2.0) * tf.reduce_prod(
            tf.cast(action_shape, tf.float32))

        terminals = tf.cast(terminals, next_values.dtype)

        # \hat Q in Equation 11 in [1]:
        Q_targets = td_targets(
            rewards=self._reward_scale * rewards,
            discounts=self._discount,
            next_values=(1 - terminals) * next_values)

        tf.debugging.assert_shapes((
            (rewards, ('B', 1)),
            (Q_targets, ('B', 1)),
            (next_values, ('B', 1)),
            (min_Q_next_target, ('B', self._value_n_particles))
        ))

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_Q(self,
                  observations,
                  actions,
                  next_observations,
                  rewards,
                  terminals):
        """Create a minimization operation for Q-function update."""
        Q_targets = self._compute_Q_targets(
            next_observations, actions, rewards, terminals)

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Q_observations = {
            name: observations[name]
            for name in self._Qs[0].observation_keys
        }

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(Q_observations, actions)
                Q_losses = (
                    0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))

            gradients = tape.gradient(Q_losses, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, observations):
        """Create a minimization operation for policy update (SVGD)."""

        expanded_observations = type(observations)((
            (key, tf.reshape(
                tf.tile(observation[:, tf.newaxis, :],
                        (1, self._kernel_n_particles, 1)),
                (-1, *observation.shape[1:])))
            for key, observation in observations.items()
        ))
        with tf.GradientTape() as tape1:
            actions = self._policy.actions(expanded_observations)
            action_shape = actions.shape[1:]
            actions = tf.reshape(
                actions, (-1, self._kernel_n_particles, *action_shape))

            tf.debugging.assert_shapes([
                (actions, (None, self._kernel_n_particles, *action_shape))])
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
        tf.debugging.assert_shapes((
            (fixed_actions, (None, n_fixed_actions, *action_shape)),
            (updated_actions, (None, n_updated_actions, *action_shape))))

        Q_observations = {
            name: tf.reshape(
                tf.tile(observations[name][:, tf.newaxis, :],
                        (1, n_fixed_actions, 1)),
                (-1, *observations[name].shape[1:]))
            for name in self._policy.observation_keys
        }
        # Target log-density. Q_soft in Equation 13:
        assert self._policy._squash
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(fixed_actions)
            Q_actions = tf.reshape(fixed_actions, (-1, *action_shape))
            Q_log_targets = tuple(
                Q.values(Q_observations, Q_actions) for Q in self._Qs)
            min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)
            svgd_target_values = tf.reshape(
                min_Q_log_target, (-1, n_fixed_actions, 1))

            squash_correction = tf.reduce_sum(
                tf.math.log(1.0 - fixed_actions ** 2 + EPS), axis=-1,
                keepdims=True)

            log_probs = svgd_target_values + squash_correction

        grad_log_probs = tape2.gradient(
            log_probs, fixed_actions)[:, :, tf.newaxis, ...]
        # grad_log_probs = tf.expand_dims(grad_log_probs, axis=2)
        grad_log_probs = tf.stop_gradient(grad_log_probs)

        tf.debugging.assert_shapes([
            (fixed_actions, (None, n_fixed_actions, *action_shape))])
        tf.debugging.assert_shapes([
            (grad_log_probs, (None, n_fixed_actions, 1, *action_shape))])

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = kernel_dict["output"][..., tf.newaxis]
        tf.debugging.assert_shapes([
            (kappa, (None, n_fixed_actions, n_updated_actions, 1))])

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_probs + kernel_dict["gradient"], axis=1)
        tf.debugging.assert_shapes([
            (action_gradients, (None, n_updated_actions, *action_shape))])

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tape1.gradient(
            updated_actions,
            self._policy.trainable_variables,
            output_gradients=action_gradients)

        with tf.GradientTape() as tape3:
            surrogate_loss = -1.0 * tf.reduce_sum([
                tf.reduce_sum(w * tf.stop_gradient(g))
                for w, g in zip(self._policy.trainable_variables, gradients)
            ])

        loss_gradients = tape3.gradient(
            surrogate_loss, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            loss_gradients, self._policy.trainable_variables))

        return surrogate_loss

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        Qs_values, Qs_losses = self._update_Q(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            batch['rewards'],
            batch['terminals'])

        policy_losses = self._update_policy(batch['observations'])

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        training_diagnostics = self._do_updates(batch)
        if iteration % self._Q_target_update_interval == 0 and self._train_Q:
            # Run target ops here.
            self._update_target(tau=tf.constant(self._tau))

        return training_diagnostics

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        diagnostics = OrderedDict()
        policy_diagnostics = self._policy.get_diagnostics({
            name: batch['observations'][name]
            for name in self._policy.observation_keys
        })
        diagnostics = OrderedDict((
            ('policy', policy_diagnostics),
        ))

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
