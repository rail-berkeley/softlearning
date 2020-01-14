from copy import copy
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.utils.gym import is_continuous_space, is_discrete_space
from .rl_algorithm import RLAlgorithm


@tf.function(experimental_relax_shapes=True)
def td_targets(rewards, discounts, next_values):
    return rewards + discounts * next_values


@tf.function(experimental_relax_shapes=True)
def compute_Q_targets(next_Q_values,
                      log_pis,
                      rewards,
                      terminals,
                      discount,
                      entropy_scale,
                      reward_scale):
    next_values = next_Q_values - entropy_scale * log_pis
    terminals = tf.cast(terminals, next_values.dtype)

    Q_targets = td_targets(
        rewards=reward_scale * rewards,
        discounts=discount,
        next_values=(1.0 - terminals) * next_values)

    return Q_targets


@tf.function(experimental_relax_shapes=True)
def _compute_Q_values(Q):
    pass


def heuristic_target_entropy(action_space):
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(action_space.shape)
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "TODO(hartikainen): implement for discrete spaces.")
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            plotter=None,

            lr=3e-4,
            alpha_lr=3e-3,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(copy(Q) for Q in Qs)

        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr
        self._alpha_lr = alpha_lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            heuristic_target_entropy(self._training_environment.action_space)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._save_full_state = save_full_state

        self._Q_optimizers = tuple(
            tf.optimizers.Adam(
                learning_rate=self._Q_lr,
                name=f'Q_{i}_optimizer'
            ) for i, Q in enumerate(self._Qs))

        self._alpha = tf.Variable(1.0, name='alpha', dtype=tf.float32)
        self._alpha_optimizer = tf.optimizers.Adam(
            self._alpha_lr, name='alpha_optimizer')

        self._policy_optimizer = tf.optimizers.Adam(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, next_observations, rewards, terminals):
        entropy_scale = self._alpha
        reward_scale = self._reward_scale
        discount = self._discount

        actions = self._policy.actions(next_observations)
        log_pis = self._policy.log_pis(next_observations, actions)
        next_Qs_values = tuple(
            Q.values(next_observations, actions) for Q in self._Q_targets)
        next_Q_values = tf.reduce_min(next_Qs_values, axis=0)

        Q_targets = compute_Q_targets(
            next_Q_values,
            log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)

        return tf.stop_gradient(Q_targets)

    @tf.function(experimental_relax_shapes=True)
    def _update_critic(self,
                       observations,
                       actions,
                       next_observations,
                       rewards,
                       terminals):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._compute_Q_targets(
            next_observations, rewards, terminals)

        tf.debugging.assert_shapes((
            (Q_targets, ('B', 1)), (rewards, ('B', 1))))

        Qs_values = []
        Qs_losses = []
        for Q, optimizer in zip(self._Qs, self._Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values = Q.values(observations, actions)
                Q_losses = (
                    0.5 * tf.losses.MSE(y_true=Q_targets, y_pred=Q_values))

            gradients = tape.gradient(Q_losses, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))
            Qs_losses.append(Q_losses)
            Qs_values.append(Q_values)

        return Qs_values, Qs_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_actor(self, observations):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        with tf.GradientTape() as tape:
            actions = self._policy.actions(observations)
            log_pis = self._policy.log_pis(observations, actions)

            Qs_log_targets = tuple(
                Q.values(observations, actions) for Q in self._Qs)
            Q_log_targets = tf.reduce_min(Qs_log_targets, axis=0)

            policy_losses = self._alpha * log_pis - Q_log_targets

        tf.debugging.assert_shapes((
            (actions, ('B', 'nA')),
            (log_pis, ('B', 1)),
            (policy_losses, ('B', 1)),
        ))

        policy_gradients = tape.gradient(
            policy_losses, self._policy.trainable_variables)

        self._policy_optimizer.apply_gradients(zip(
            policy_gradients, self._policy.trainable_variables))

        return policy_losses

    @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, observations):
        if not isinstance(self._target_entropy, Number):
            return

        actions = self._policy.actions(observations)
        log_pis = self._policy.log_pis(observations, actions)

        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                self._alpha * tf.stop_gradient(log_pis + self._target_entropy))
            # NOTE(hartikainen): It's important that we take the average here,
            # otherwise we end up effectively having `batch_size` times too
            # large learning rate.
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self._alpha])
        self._alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self._alpha]))

        return alpha_losses

    def _init_training(self):
        self._update_target(tau=tf.constant(1.0))

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self._Qs, self._Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

    @tf.function(experimental_relax_shapes=True)
    def _do_updates(self, batch):
        Qs_values, Qs_losses = self._update_critic(
            batch['observations'],
            batch['actions'],
            batch['next_observations'],
            batch['rewards'],
            batch['terminals'])

        policy_losses = self._update_actor(batch['observations'])
        alpha_losses = self._update_alpha(batch['observations'])

        diagnostics = OrderedDict((
            ('Q_value-mean', tf.reduce_mean(Qs_values)),
            ('Q_loss-mean', tf.reduce_mean(Qs_losses)),
            ('policy_loss-mean', tf.reduce_mean(policy_losses)),
            ('alpha', self._alpha),
            ('alpha_loss-mean', tf.reduce_mean(alpha_losses)),
        ))
        return diagnostics

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        training_diagnostics = self._do_updates(batch)

        if iteration % self._target_update_interval == 0:
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

        diagnostics = OrderedDict((
            ('alpha', self._alpha.numpy()),
            ('policy', self._policy.get_diagnostics(
                batch['observations'])),
        ))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_alpha': self._alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
