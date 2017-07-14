from __future__ import absolute_import

import abc
import gtimer as gt

import numpy as np

from softqlearning.misc.replay_pool import SimpleReplayPool, DoublePool
from softqlearning.misc import tf_utils

from rllab.algos.base import RLAlgorithm
from softqlearning.misc import logger


class OnlineAlgorithm(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            batch_size=64,
            n_epochs=1000,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            max_path_length=1000,
            scale_reward=1.,
            render=False,
            demo_pool=None,
            demo_ratio=0.1
    ):
        """
        :param batch_size: Minibatch size for training.
        :param n_epochs: Number of epochs.
        :param epoch_length: Number of time steps per epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the replay pool.
        :param max_path_length: Maximum episode length.
        :param scale_reward: Rewards multiplier.
        :param render: Boolean. If True, render the environment.
        """
        assert min_pool_size >= 2

        self._replay_pool_size = replay_pool_size
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._epoch_length = epoch_length
        self._min_pool_size = min_pool_size
        self._max_path_length = max_path_length
        self._scale_reward = scale_reward
        self._render = render
        self._demo_pool = demo_pool
        self._demo_ratio = demo_ratio
        self._double_pool = None

        self._sess = tf_utils.create_session()

    def _train(self, env, policy):
        self._init_training(env, policy)

        with self._sess.as_default():
            observation = env.reset()
            policy.reset()
            itr = 0
            path_length = 0
            path_return = 0
            gt.rename_root('online algo')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    # Sample next action and state.
                    action, _ = policy.get_action(observation)
                    gt.stamp('train: get actions')
                    action.squeeze()
                    if self._render:
                        env.render()
                    next_ob, raw_reward, terminal, info = env.step(action)
                    reward = raw_reward * self._scale_reward
                    path_length += 1
                    path_return += reward
                    gt.stamp('train: simulation')

                    # Add experience to replay pool.
                    self._pool.add_sample(observation,
                                          action,
                                          reward,
                                          terminal,
                                          False)
                    should_reset = (terminal or
                                    path_length >= self._max_path_length)
                    if should_reset:
                        # noinspection PyTypeChecker
                        self._pool.add_sample(next_ob,
                                              np.zeros_like(action),
                                              np.zeros_like(reward),
                                              np.zeros_like(terminal),
                                              True)

                        observation = env.reset()
                        policy.reset()
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob
                    gt.stamp('train: fill replay pool')

                    # Train.
                    if self._pool.size >= self._min_pool_size:
                        self._do_training(itr)
                    itr += 1
                    gt.stamp('train: updates')

                # Evaluate.
                self._evaluate(epoch)
                gt.stamp("test")

                # Log.
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                train_time = np.sum([
                    times_itrs[key][-1] for key in times_itrs.keys()
                    if 'train: ' in key
                ])

                eval_time = times_itrs["test"][-1]
                total_time = gt.get_times().total
                logger.record_tabular("time: train", train_time)
                logger.record_tabular("time: eval", eval_time)
                logger.record_tabular("time: total", total_time)
                logger.record_tabular("scale_reward", self._scale_reward)
                logger.record_tabular("epochs", epoch)
                logger.record_tabular("steps: all", itr)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                gt.stamp("logging")

                print(gt.report(
                    include_itrs=False,
                    format_options={
                        'itr_name_width': 30
                    },
                ))

            env.terminate()

    @gt.wrap
    def _do_training(self, itr):
        if self._double_pool:
            minibatch = self._double_pool.random_batch(self._batch_size)
        else:
            minibatch = self._pool.random_batch(self._batch_size)
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards']
        sampled_next_obs = minibatch['next_observations']
        gt.stamp("train: sample minibatch")

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)
        gt.stamp("train: update feed dict")

        training_ops = self._get_training_ops(itr)
        gt.stamp("train: get training ops")

        self._sess.run(training_ops, feed_dict)
        gt.stamp("train: run training ops")

        target_ops = self._get_target_ops(itr)
        gt.stamp("train: get target ops")

        self._sess.run(target_ops, feed_dict)
        gt.stamp("train: run target ops")

    @abc.abstractmethod
    def get_epoch_snapshot(self, epoch):
        return

    def _init_training(self, env, policy):
        """ Method to be called at the start of training.

        :param env: Environment instance.
        :param policy:  Policy instance.
        :return: None
        """
        observation_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim
        self._pool = SimpleReplayPool(self._replay_pool_size,
                                      observation_dim,
                                      action_dim)

        if self._demo_pool:
            self._double_pool = DoublePool(
                self._demo_pool, self._pool, self._demo_ratio
            )

    @abc.abstractmethod
    def _get_training_ops(self, itr):
        """ Returns training operations.

        :param itr: Iteration number.
        :return: List of ops to perform when training.
        """
        return

    @abc.abstractmethod
    def _get_target_ops(self, itr):
        """ Returns operations for updating target networks (if any).

        :param itr: Iteration number.
        :return: List of ops to perform when training.
        """
        return

    @abc.abstractmethod
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        """
        :param rewards: Minibatch of rewards.
        :param terminals: Minibatch of terminal variables.
        :param obs: Minibatch of observations.
        :param actions: Minibatch of actions.
        :param next_obs: Minibatch of observations at the next time step.
        :return: Dictionary needed for the ops returned by get_training_ops.
        """
        return

    @abc.abstractmethod
    def _evaluate(self, epoch):
        """ Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """
        return
