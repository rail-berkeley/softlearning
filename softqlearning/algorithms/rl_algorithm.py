import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from softqlearning.misc.utils import deep_clone
from softqlearning.misc import tf_utils
from softqlearning.misc.sampler import rollouts


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_render=False,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length

        self._eval_n_episodes = eval_n_episodes
        self._eval_render = eval_render

        self.env = None
        self.policy = None
        self.pool = None

    def _train(self, env, policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._init_training()
        self.sampler.initialize(env, policy, pool)

        evaluation_env = deep_clone(env) if self._eval_n_episodes else None

        with tf_utils.get_default_session().as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                self._evaluate(policy, evaluation_env)
                gt.stamp('eval')

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs['eval'][-1]
                time_total = gt.get_times().total
                time_train = time_itrs.get('train', [0])[-1]
                time_sample = time_itrs.get('sample', [0])[-1]

                logger.record_tabular('time-train', time_train)
                logger.record_tabular('time-eval', time_eval)
                logger.record_tabular('time-sample', time_sample)
                logger.record_tabular('time-total', time_total)
                logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

            self.sampler.terminate()

    def _evaluate(self, policy, evaluation_env):
        """Perform evaluation for the current policy."""

        if self._eval_n_episodes < 1:
            return

        # TODO: max_path_length should be a property of environment.
        paths = rollouts(evaluation_env, policy, self.sampler._max_path_length,
                         self._eval_n_episodes)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        evaluation_env.log_diagnostics(paths)
        if self._eval_render:
            evaluation_env.render(paths)

        if self.sampler.batch_ready():
            batch = self.sampler.random_batch()
            self.log_diagnostics(batch)

    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self):
        raise NotImplementedError
