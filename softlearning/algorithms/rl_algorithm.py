import abc
import gtimer as gt
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from rllab.algos.base import Algorithm

from softlearning.samplers import rollouts, rollout


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            n_initial_exploration_steps=0,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        if control_interval != 1:
            # TODO(hartikainen): we used to support control_interval in our old
            # SAC code, but it was removed because of the hacky implementation.
            # This functionality should be implemented as a part of the
            # sampler. See `sac/algos/base.py@fe7dc7c` for the old
            # implementation.
            raise NotImplementedError(
                "Control interval has been temporarily removed. See the"
                " comments in RlAlgorithm.__init__ for more information.")

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf.keras.backend.get_session()

        self._env = None
        self._policy = None
        self._pool = None

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        for i in range(self._n_initial_exploration_steps):
            self.sampler.sample()

    def _training_before_hook(self):
        """Method called before the actual training loops."""
        pass

    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _epoch_before_hook(self, epoch):
        """Hook called at the beginning of each epoch."""
        pass

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    def _training_batch(self, batch_size=None):
        return self.sampler.random_batch(batch_size)

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    def _train(self, env, policy, pool, initial_exploration_policy=None):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training()

        self._initial_exploration_hook(env, initial_exploration_policy, pool)
        self.sampler.initialize(env, policy, pool)

        self._training_before_hook()

        evaluation_env = env.copy() if self._eval_n_episodes else None

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                self._epoch_before_hook(epoch)

                for t in range(1, self._epoch_length + 1):
                    self._do_sampling(epoch=epoch, epoch_timestep=t)
                    gt.stamp('sample')
                    if self.ready_to_train:
                        self._do_training_repeats(epoch=epoch,
                                                  epoch_timestep=t)
                        gt.stamp('train')

                evaluation_diagnostics = self._evaluate(policy, evaluation_env, epoch)
                gt.stamp('eval')

                training_paths = self.sampler.get_last_n_paths()
                self._epoch_after_hook(epoch, training_paths)

                params = self.get_snapshot(epoch)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs.get('eval', [0])[-1]
                time_total = gt.get_times().total
                time_train = time_itrs.get('train', [0])[-1]
                time_sample = time_itrs.get('sample', [0])[-1]

                diagnostics = OrderedDict({
                    'time-train': time_train,
                    'time-eval': time_eval,
                    'time-sample': time_sample,
                    'time-total': time_total,
                    'timesteps_total': epoch * self._epoch_length + t
                })

                sampler_diagnostics = self.sampler.get_diagnostics()
                diagnostics.update({
                    f'sampler/{key}': value
                    for key, value in sampler_diagnostics.items()
                })

                diagnostics.update({
                    f'evaluation/{key}': value
                    for key, value in evaluation_diagnostics.items()
                })

                yield epoch, diagnostics

            self.sampler.terminate()

        self._training_after_hook()

    def _evaluate(self, policy, evaluation_env, epoch):
        """Perform evaluation for the current policy."""

        if self._eval_n_episodes < 1:
            return

        if hasattr(policy, 'deterministic'):
            with policy.deterministic(self._eval_deterministic):
                # TODO: max_path_length should be a property of environment.
                paths = rollouts(evaluation_env,
                                 policy,
                                 self.sampler._max_path_length,
                                 self._eval_n_episodes,
                                 render=self._eval_render)
        else:
            paths = rollouts(evaluation_env, policy,
                             self.sampler._max_path_length,
                             self._eval_n_episodes)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        iteration = epoch * self._epoch_length
        batch = self._evaluation_batch()

        diagnostics = self.get_diagnostics(iteration, batch, paths)
        diagnostics.update({
            'return-average': np.mean(total_returns),
            'return-min': np.min(total_returns),
            'return-max': np.max(total_returns),
            'return-std': np.std(total_returns),
            'episode-length-avg': np.mean(episode_lengths),
            'episode-length-min': np.min(episode_lengths),
            'episode-length-max': np.max(episode_lengths),
            'episode-length-std': np.std(episode_lengths),
        })

        env_infos = evaluation_env.get_path_infos(paths)
        for key, value in env_infos.items():
            diagnostics[f'evaluation_env/{key}'] = value

        if self._eval_render and hasattr(evaluation_env, 'render_rollouts'):
            # TODO(hartikainen): Make this consistent such that there's no need
            # for the hasattr check.
            evaluation_env.render_rollouts(paths)

        return diagnostics

    @abc.abstractmethod
    def get_diagnostics(self, iteration, batch, paths):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        return self.sampler.batch_ready()

    def _do_sampling(self, epoch, epoch_timestep):
        self.sampler.sample()

    def _do_training_repeats(self, epoch, epoch_timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        total_timestep = epoch * self._epoch_length + epoch_timestep
        if total_timestep % self._train_every_n_steps > 0: return

        for i in range(self._n_train_repeat):
            self._do_training(
                iteration=total_timestep,
                batch=self._training_batch())

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError
