"""
:author: Vitchyr Pong
"""
import abc
import pickle
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import tensorflow as tf

from railrl.policies.nn_policy import NNPolicy
from railrl.core.neuralnet import NeuralNetwork
from railrl.data_management.simple_replay_pool import SimpleReplayPool
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler


class OnlineAlgorithm(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            policy: NNPolicy,
            exploration_strategy,
            batch_size=64,
            n_epochs=1000,
            epoch_length=10000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            soft_target_tau=1e-2,
            max_path_length=1000,
            eval_samples=10000,
            scale_reward=1.,
            render=False,
            n_updates_per_time_step=1,
            batch_norm_config=None,
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: A Policy
        :param replay_pool_size: Size of the replay pool
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param min_pool_size: Minimum size of the pool to start training.
        :param discount: Discount factor for the MDP
        :param soft_target_tau: Moving average rate. 1 = update immediately
        :param max_path_length: Maximum path length
        :param eval_samples: Number of time steps to take for evaluation.
        :param scale_reward: How much to multiply the rewards by.
        :param render: Boolean. If True, render the environment.
        :param n_updates_per_time_step: How many SGD steps to take per time
        step.
        :param batch_norm_config: Config for batch_norm. If set, batch_norm
        is enabled.
        :return:
        """
        assert min_pool_size >= 2
        # Have two separate env's to make sure that the training and eval
        # envs don't affect one another.
        self.training_env = env
        self.env = pickle.loads(pickle.dumps(self.training_env))
        self.policy = policy
        self.exploration_strategy = exploration_strategy
        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.scale_reward = scale_reward
        self.render = render
        self.n_updates_per_time_step = n_updates_per_time_step
        self._batch_norm = batch_norm_config is not None
        self._batch_norm_config = batch_norm_config

        self.observation_dim = self.training_env.observation_space.flat_dim
        self.action_dim = self.training_env.action_space.flat_dim
        self.rewards_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 1],
                                                  name='rewards')
        self.terminals_placeholder = tf.placeholder(tf.float32,
                                                    shape=[None, 1],
                                                    name='terminals')
        self.pool = SimpleReplayPool(self.replay_pool_size,
                                     self.observation_dim,
                                     self.action_dim)
        self.last_statistics = None
        self.sess = tf.get_default_session() or tf.Session()
        with self.sess.as_default():
            self._init_tensorflow_ops()
        self.es_path_returns = []

        self.eval_sampler = BatchSampler(self)
        self.scope = None  # Necessary for BatchSampler
        self.whole_paths = True  # Also for BatchSampler

    def _start_worker(self):
        self.eval_sampler.start_worker()

    def _shutdown_worker(self):
        self.eval_sampler.shutdown_worker()

    def _sample_paths(self, epoch):
        # Sampler uses self.batch_size to figure out how many samples to get
        saved_batch_size = self.batch_size
        self.batch_size = self.n_eval_samples
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
            max_path_length=self.max_path_length,
            batch_size=self.n_eval_samples
        )
        self.batch_size = saved_batch_size
        return paths

    @overrides
    def train(self):
        with self.sess.as_default():
            self._init_training()
            self._start_worker()
            self._switch_to_training_mode()

            observation = self.training_env.reset()
            self.exploration_strategy.reset()
            itr = 0
            path_length = 0
            path_return = 0
            for epoch in range(self.n_epochs):
                logger.push_prefix('Epoch #%d | ' % epoch)
                logger.log("Training started")
                start_time = time.time()
                for _ in range(self.epoch_length):
                    with self._eval_then_training_mode():
                        action = self.exploration_strategy.get_action(itr,
                                                                      observation,
                                                                      self.policy)
                    if self.render:
                        self.training_env.render()
                    next_ob, raw_reward, terminal, _ = self.training_env.step(
                        self.process_action(action)
                    )
                    # Some envs return a Nx1 vector for the observation
                    next_ob = next_ob.flatten()
                    reward = raw_reward * self.scale_reward
                    path_length += 1
                    path_return += reward

                    self.pool.add_sample(observation,
                                         action,
                                         reward,
                                         terminal,
                                         False)
                    if terminal or path_length >= self.max_path_length:
                        self.pool.add_sample(next_ob,
                                             np.zeros_like(action),
                                             np.zeros_like(reward),
                                             np.zeros_like(terminal),
                                             True)
                        observation = self.training_env.reset()
                        self.exploration_strategy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob

                    if self.pool.size >= self.min_pool_size:
                        for _ in range(self.n_updates_per_time_step):
                            self._do_training()
                    itr += 1

                logger.log("Training finished. Time: {0}".format(time.time() -
                                                                 start_time))
                with self._eval_then_training_mode():
                    if self.pool.size >= self.min_pool_size:
                        start_time = time.time()
                        if self.n_eval_samples > 0:
                            self.evaluate(epoch, self.es_path_returns)
                            self.es_path_returns = []
                        params = self.get_epoch_snapshot(epoch)
                        logger.log(
                            "Eval time: {0}".format(time.time() - start_time))
                        logger.save_itr_params(epoch, params)
                    logger.dump_tabular(with_prefix=False)
                    logger.pop_prefix()
            self._switch_to_eval_mode()
            self.training_env.terminate()
            self._shutdown_worker()
            return self.last_statistics

    def _switch_to_training_mode(self):
        """
        Make any updates needed so that the internal networks are in training
        mode.
        :return:
        """
        for network in self._networks:
            network.switch_to_training_mode()

    def _switch_to_eval_mode(self):
        """
        Make any updates needed so that the internal networks are in eval mode.
        :return:
        """
        for network in self._networks:
            network.switch_to_eval_mode()

    @contextmanager
    def _eval_then_training_mode(self):
        """
        Helper method to quickly switch to eval mode and then to training mode

        ```
        # doesn't matter what mode you were in
        with self.eval_then_training_mode():
            # in eval mode
        # in training mode
        :return:
        """
        self._switch_to_eval_mode()
        yield
        self._switch_to_training_mode()

    def _do_training(self):
        minibatch = self.pool.random_batch(self.batch_size)
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards']
        sampled_next_obs = minibatch['next_observations']

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)
        ops = self._get_training_ops()
        if isinstance(ops[0], list):
            for op in ops:
                self.sess.run(op, feed_dict=feed_dict)
        else:
            self.sess.run(ops, feed_dict=feed_dict)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
        )

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

    @property
    @abc.abstractmethod
    def _networks(self) -> Iterable[NeuralNetwork]:
        """
        :return: List of networks used in the algorithm.

        It's crucial that this list is up to date!
        """
        pass


    @abc.abstractmethod
    def _init_tensorflow_ops(self):
        """
        Method to be called in the initialization of the class. After this
        method is called, the train() method should work.
        :return: None
        """
        return

    @abc.abstractmethod
    def _init_training(self):
        """
        Method to be called at the start of training.
        :return: None
        """
        return

    @abc.abstractmethod
    def _get_training_ops(self):
        """
        :return: List of ops to perform when training. If a list of list is
        provided, each list is executed in order with separate calls to
        sess.run.
        """
        return

    @abc.abstractmethod
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        """
        :return: feed_dict needed for the ops returned by get_training_ops.
        """
        return

    @abc.abstractmethod
    def evaluate(self, epoch, es_path_returns):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param es_path_returns: List of path returns from explorations strategy
        :return: Dictionary of statistics.
        """
        return

    def process_action(self, raw_action):
        """
        Process the action outputted by the policy before giving it to the
        environment.

        :param raw_action:
        :return:
        """
        return raw_action
