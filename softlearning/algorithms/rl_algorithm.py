import abc
from collections import OrderedDict
from distutils.version import LooseVersion
from itertools import count
import gtimer as gt
import math
import os

import numpy as np
import tensorflow as tf
import tree

from softlearning.samplers import rollouts
from softlearning.utils.video import save_video
from softlearning.policies import utils as policy_utils


if LooseVersion(tf.__version__) > LooseVersion("2.00"):
    from tensorflow.python.training.tracking.tracking import (
        AutoTrackable as Checkpointable)
else:
    from tensorflow.contrib.checkpoint import Checkpointable


class RLAlgorithm(Checkpointable):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            pool,
            sampler,
            n_epochs=1000,
            train_every_n_steps=1,
            n_train_repeat=1,
            min_pool_size=1,
            batch_size=1,
            max_train_repeat_per_timestep=5,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render_kwargs=None,
            video_save_frequency=0,
            num_warmup_samples=0,
    ):
        """
        Args:
            pool (`ReplayPool`): Replay pool to add gathered samples to.
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_render_kwargs (`None`, `dict`): Arguments to be passed for
                rendering evaluation rollouts. `None` to disable rendering.
            num_warmup_samples ('int'): Number of random samples to warmup the
                replay pool with.
        """
        self.sampler = sampler
        self.pool = pool

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._max_train_repeat_per_timestep = max(
            max_train_repeat_per_timestep, n_train_repeat)
        self._train_every_n_steps = train_every_n_steps
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes
        self._video_save_frequency = video_save_frequency
        self._num_warmup_samples = num_warmup_samples

        self._eval_render_kwargs = eval_render_kwargs or {}

        if self._video_save_frequency > 0:
            render_mode = self._eval_render_kwargs.pop('mode', 'rgb_array')
            assert render_mode != 'human', (
                "RlAlgorithm cannot render and save videos at the same time")
            self._eval_render_kwargs['mode'] = render_mode

        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0

    def _do_warmup_samples(self):
        num_warmup_samples = self._num_warmup_samples - self.pool.size
        if num_warmup_samples < 1:
            return

        uniform_policy = policy_utils.get_uniform_policy(
            self.sampler.environment)

        old_policy = self.sampler.policy
        self.sampler.policy = uniform_policy
        while self.pool.size < num_warmup_samples:
            self.sampler.sample()
        self.sampler.policy = old_policy

    def _training_before_hook(self):
        """Method called before the actual training loops."""
        self._do_warmup_samples()

    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass

    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass

    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    def _training_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        return self.pool.random_batch(batch_size, **kwargs)

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    @property
    def _training_started(self):
        return self._total_timestep > 0

    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        return self._train(*args, **kwargs)

    def _train(self):
        """Return a generator that runs the standard RL loop."""
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            update_diagnostics = []

            start_samples = self.sampler._total_samples
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                if self.ready_to_train:
                    repeat_diagnostics = self._do_training_repeats(
                        timestep=self._total_timestep)
                    if repeat_diagnostics is not None:
                        update_diagnostics.append(repeat_diagnostics)

                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            update_diagnostics = tree.map_structure(
                lambda *d: np.mean(d), *update_diagnostics)

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths,
                training_environment,
                self._total_timestep,
                evaluation_type='train')
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths,
                    evaluation_environment,
                    self._total_timestep,
                    evaluation_type='evaluation')
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = {
                key: times[-1]
                for key, times in gt.get_times().stamps.itrs.items()
            }

            # TODO(hartikainen/tf2): Fix the naming of training/update
            # diagnostics/metric
            diagnostics.update((
                ('evaluation', evaluation_metrics),
                ('training', training_metrics),
                ('update', update_diagnostics),
                ('times', time_diagnostics),
                ('sampler', sampler_diagnostics),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('total_timestep', self._total_timestep),
                ('num_train_steps', self._num_train_steps),
            ))

            if self._eval_render_kwargs and hasattr(
                    evaluation_environment, 'render_rollouts'):
                # TODO(hartikainen): Make this consistent such that there's no
                # need for the hasattr check.
                training_environment.render_rollouts(evaluation_paths)

            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        yield {'done': True, **diagnostics}

    def _evaluation_paths(self, policy, evaluation_env):
        if self._eval_n_episodes < 1: return ()

        # TODO(hartikainen): I don't like this way of handling evaluation mode
        # for the policies. We should instead have two separete policies for
        # training and evaluation.
        with policy.evaluation_mode():
            paths = rollouts(
                self._eval_n_episodes,
                evaluation_env,
                policy,
                self.sampler._max_path_length,
                render_kwargs=self._eval_render_kwargs)

        should_save_video = (
            self._video_save_frequency > 0
            and (self._epoch == 0
                 or (self._epoch + 1) % self._video_save_frequency == 0))

        if should_save_video:
            fps = 1 // getattr(self._training_environment, 'dt', 1/30)
            for i, path in enumerate(paths):
                video_frames = path.pop('images')
                video_file_name = f'evaluation_path_{self._epoch}_{i}.mp4'
                video_file_path = os.path.join(
                    os.getcwd(), 'videos', video_file_name)
                save_video(video_frames, video_file_path, fps=fps)

        return paths

    def _evaluate_rollouts(self,
                           episodes,
                           environment,
                           timestep,
                           evaluation_type=None):
        """Compute evaluation metrics for the given rollouts."""

        episodes_rewards = [episode['rewards'] for episode in episodes]
        episodes_reward = [np.sum(episode_rewards)
                           for episode_rewards in episodes_rewards]
        episodes_length = [episode_rewards.shape[0]
                           for episode_rewards in episodes_rewards]

        diagnostics = OrderedDict((
            ('episode-reward-mean', np.mean(episodes_reward)),
            ('episode-reward-min', np.min(episodes_reward)),
            ('episode-reward-max', np.max(episodes_reward)),
            ('episode-reward-std', np.std(episodes_reward)),
            ('episode-length-mean', np.mean(episodes_length)),
            ('episode-length-min', np.min(episodes_length)),
            ('episode-length-max', np.max(episodes_length)),
            ('episode-length-std', np.std(episodes_length)),
        ))

        environment_infos = environment.get_path_infos(
            episodes, timestep, evaluation_type=evaluation_type)
        diagnostics['environment_infos'] = environment_infos

        return diagnostics

    @abc.abstractmethod
    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        return self._min_pool_size <= self.pool.size

    def _do_sampling(self, timestep):
        self.sampler.sample()

    def _do_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        diagnostics = [
            self._do_training(iteration=timestep, batch=self._training_batch())
            for i in range(self._n_train_repeat)
        ]

        diagnostics = tree.map_structure(
            lambda *d: tf.reduce_mean(d).numpy(), *diagnostics)

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

        return diagnostics

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    def _init_training(self):
        pass

    @property
    def tf_saveables(self):
        return {}

    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
            '_num_train_steps': self._num_train_steps,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
