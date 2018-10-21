from numbers import Number

import numpy as np
import tensorflow as tf

from serializable import Serializable
from rllab.misc import logger

from .rl_algorithm import RLAlgorithm


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm, Serializable):
    """Soft Actor-Critic (SAC)

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
            initial_exploration_policy,
            Qs,
            V,
            pool,
            plotter=None,
            tf_summaries=False,

            lr=3e-3,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            summary_dir=None,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            V: Soft value function approximator.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        self._Serializable__initialize(locals())

        self.global_step = tf.get_variable(
            "global_step",
            (),
            trainable=False,
            dtype=tf.int64,
            initializer=tf.constant_initializer(0, dtype=tf.int64))

        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._initial_exploration_policy = initial_exploration_policy
        self._Qs = Qs
        self._V = V
        self._V_target = tf.keras.models.clone_model(self._V)
        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr
        self._V_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._env.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.
        self._reparameterize = self._policy._reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        self._save_full_state = save_full_state

        observation_shape = self._env.active_observation_shape
        action_shape = self._env.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._Do = observation_shape[0]
        assert len(action_shape) == 1, action_shape
        self._Da = action_shape[0]

        self._build()
        # TODO(hartikainen): Should the _initialize_tf_variables call happen
        # outside of this class/method?
        self._initialize_tf_variables()

    def _build(self):
        self._training_ops = {}
        self._target_update_ops = {}
        self._summary_ops = {}

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_summary_ops()

    def _initialize_tf_variables(self):
        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and Q and V variables. tf.metrics (used at
        # least in the LFP-policy) uses local variables.
        uninit_vars = []
        for var in tf.global_variables() + tf.local_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)

        self._sess.run(tf.variables_initializer(uninit_vars))

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""

        return self._train(
            self._env,
            self._policy,
            self._pool,
            initial_exploration_policy=self._initial_exploration_policy,
            *args,
            **kwargs)

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, ),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, self._Da),
                name='raw_actions',
            )

    def _get_Q_target(self):
        V_next_target = self._V_target(self._next_observations_ph)

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * V_next_target
        )  # N

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=tf.train.AdamOptimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                name="{}_optimizer".format(Q._name),
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for Q, Q_loss in zip(self._Qs, Q_losses))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

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

        if (getattr(self._policy, '_observations_preprocessor', False)
            and self._tf_summaries):
            self.embeddings = self._policy._observations_preprocessor(
                self._observations_ph)
            tf.contrib.layers.summarize_activation(self.embeddings)

        actions, log_pi = self._policy.actions_for(
            observations=self._observations_ph, with_log_pis=True, reuse=True)

        log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pi + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        V_value = self._V_value = self._V(self._observations_ph)  # N

        if self._action_prior == 'normal':
            D_s = actions.shape.as_list()[-1]
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_loss = tf.reduce_mean(
                alpha * log_pi - min_Q_log_target - policy_prior_log_probs)
        else:
            policy_kl_loss = tf.reduce_mean(
                log_pi * tf.stop_gradient(
                    alpha * log_pi - min_Q_log_target + V_value
                    - policy_prior_log_probs))

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        policy_loss = (policy_kl_loss + policy_regularization_loss)

        # We update the V towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        V_target = tf.stop_gradient(
            min_Q_log_target
            - alpha * log_pi
            + policy_prior_log_probs)

        V_loss = self._V_loss = tf.losses.mean_squared_error(
            labels=V_target, predictions=V_value, weights=0.5)

        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=tf.train.AdamOptimizer,
            variables=self._policy.get_params_internal(),
            increment_global_step=False,
            name="policy_optimizer",
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        V_train_op = tf.contrib.layers.optimize_loss(
            V_loss,
            self.global_step,
            learning_rate=self._V_lr,
            optimizer=tf.train.AdamOptimizer,
            variables=self._V.trainable_variables,
            increment_global_step=True,
            name="V_optimizer",
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({
            'policy_train_op': policy_train_op,
            'V_train_op': V_train_op,
        })

    def _init_summary_ops(self):
        summary_dir = logger._snapshot_dir
        if self._tf_summaries and summary_dir is not None:
            # TODO(hartikainen): This should get the logdir some other way than
            # from the rllab logger.
            self.summary_writer = tf.summary.FileWriter(
                summary_dir, self._sess.graph)
            self._summary_ops.update({'all': tf.summary.merge_all()})
        else:
            self.summary_writer = None

    def _init_training(self):
        pass

    def _update_target(self):
        source_params = self._V.get_weights()
        target_params = self._V_target.get_weights()

        self._V_target.set_weights([
            (1 - self._tau) * target + self._tau * source
            for target, source in zip(target_params, source_params)
        ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)

        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def log_diagnostics(self, iteration, batch, paths):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, V_value, Q_losses,
         summary_results, alpha, global_step) = self._sess.run(
            (self._Q_values,
             self._V_value,
             self._Q_losses,
             self._summary_ops,
             self._alpha,
             self.global_step),
            feed_dict)

        if summary_results and self.summary_writer is not None:
            self.summary_writer.add_summary(
                summary_results['all'], global_step)
            self.summary_writer.flush()  # Not sure if this is needed

        logger.record_tabular('Q-avg', np.mean(Q_values))
        logger.record_tabular('Q-std', np.std(Q_values))

        logger.record_tabular('V-avg', np.mean(V_value))
        logger.record_tabular('V-std', np.std(V_value))

        logger.record_tabular('Q_loss', np.mean(Q_losses))

        logger.record_tabular('alpha', alpha)

        self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'policy': self._policy,
                'Qs': self._Qs,
                'V': self._V,
                'env': self._env,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'Qs-params': tuple(Q.get_weights() for Q in self._Qs),
            'V-params': self._V.get_weights(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)

        for i, Qs_params in enumerate(d['Qs-params']):
            self._Qs[i].set_weights(Qs_params)
        self._V.set_weights(d['V-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
