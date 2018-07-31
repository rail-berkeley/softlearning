import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from softlearning.misc.utils import deep_clone
from softlearning.misc import tf_utils
from softlearning.samplers import rollouts


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
            n_initial_exploration_steps=10000,
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

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._policy = None
        self._pool = None

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

        if initial_exploration_policy is None:
            self.sampler.initialize(env, policy, pool)
            initial_exploration_done = True
        else:
            self.sampler.initialize(env, initial_exploration_policy, pool)
            initial_exploration_done = False

        evaluation_env = deep_clone(env) if self._eval_n_episodes else None

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(
                    range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    if (not initial_exploration_done
                            and (self._epoch_length * epoch
                                 >= self._n_initial_exploration_steps)):
                        self.sampler.set_policy(policy)
                        initial_exploration_done = True
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                mean_returns = self._evaluate(policy, evaluation_env, epoch)
                gt.stamp('eval')

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)

                time_itrs = gt.get_times().stamps.itrs
                time_eval = time_itrs.get('eval', [0])[-1]
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

                yield epoch, mean_returns

            self.sampler.terminate()

    def _evaluate(self, policy, evaluation_env, epoch):
        """Perform evaluation for the current policy."""

        if self._eval_n_episodes < 1:
            return

        if hasattr(policy, 'deterministic'):
            with policy.deterministic(self._eval_deterministic):
                # TODO: max_path_length should be a property of environment.
                paths = rollouts(evaluation_env, policy,
                                 self.sampler._max_path_length,
                                 self._eval_n_episodes)
        else:
            paths = rollouts(evaluation_env, policy,
                             self.sampler._max_path_length,
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

        env_infos = evaluation_env.get_path_infos(paths)
        for key, value in env_infos.items():
            logger.record_tabular(key, value)

        if self._eval_render:
            evaluation_env.render(paths)

        iteration = epoch * self._epoch_length
        batch = self.sampler.random_batch()
        self.log_diagnostics(iteration, batch)

        return np.mean(total_returns)

    @abc.abstractmethod
    def log_diagnostics(self, iteration, batch):
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
