import abc
from collections import defaultdict
import unittest

import numpy as np
import gym
import pytest

from softlearning.replay_pools.hindsight_experience_replay_pool import (
    HindsightExperienceReplayPool)
from softlearning.environments.utils import get_environment


def create_pool(env, max_size=100, **kwargs):
    return HindsightExperienceReplayPool(
        environment=env, max_size=max_size, **kwargs)


HER_STRATEGY_TYPES = ['random', 'final', 'episode', 'future']
HER_RESAMPLING_PROBABILITIES = [0, 0.3, 0.5, 0.8, 1.0]
REWARD_FUNCTIONS = ()
TERMINAL_FUNCTIONS = ()


class StrategyValidator(object):
    def __init__(self, her_strategy):
        self._her_strategy = her_strategy
        self._statistics = defaultdict(list)

    @abc.abstractmethod
    def verify_batch(self, batch):
        where_not_resampled = np.flatnonzero(~batch['resampled'])
        np.testing.assert_equal(
            batch['resampled_distances'][where_not_resampled],
            float('inf'))

        self._statistics['num_resampled'] = batch['resampled']
        self._statistics['num_total'] = batch['resampled'].size

    def statistics_match(self):
        num_resampleds = np.array(self._statistics['num_resampled'])
        num_totals = np.array(self._statistics['num_total'])
        proportion_resampled = (
            np.sum(num_resampleds) / np.sum(num_totals))
        expected_proportion_resampled = self._her_strategy[
            'resampling_probability']
        if expected_proportion_resampled in (0, 1):
            assert proportion_resampled == expected_proportion_resampled
        else:
            assert np.abs(
                proportion_resampled - expected_proportion_resampled
            ) < 0.1

        return True


class RandomStrategyValidator(StrategyValidator):
    pass


class FinalStrategyValidator(StrategyValidator):
    def verify_batch(self, batch):
        if np.sum(batch['resampled']) > 0:
            where_resampled = np.flatnonzero(batch['resampled'])
            np.testing.assert_equal(
                batch['resampled_distances'][where_resampled],
                batch['episode_index_backwards'][where_resampled])
        super(FinalStrategyValidator, self).verify_batch(batch)


class EpisodeStrategyValidator(StrategyValidator):
    def verify_batch(self, batch):
        if np.sum(batch['resampled']) > 0:
            where_resampled = np.flatnonzero(batch['resampled'])

            gt_first_index = (
                -1 * batch['episode_index_forwards'][where_resampled]
                <= batch['resampled_distances'][where_resampled])
            lt_final_index = (
                batch['resampled_distances'][where_resampled]
                < batch['episode_index_backwards'][where_resampled])
            within_episode = np.logical_and(
                gt_first_index, lt_final_index)
            assert np.all(within_episode)
        super(EpisodeStrategyValidator, self).verify_batch(batch)


class FutureStrategyValidator(StrategyValidator):
    def verify_batch(self, batch):
        if np.sum(batch['resampled']) > 0:
            where_resampled = np.flatnonzero(batch['resampled'])
            assert np.all(batch['resampled_distances'][where_resampled] >= 0)
            assert np.all(
                batch['resampled_distances'][where_resampled]
                <= batch['episode_index_backwards'][where_resampled])
        super(FutureStrategyValidator, self).verify_batch(batch)


class TestHindsightExperienceReplayPool():
    @pytest.mark.parametrize("strategy_type", HER_STRATEGY_TYPES)
    @pytest.mark.parametrize("resampling_probability",
                             HER_RESAMPLING_PROBABILITIES)
    def test_resampling(self, strategy_type, resampling_probability):
        env = get_environment('gym', 'HandReach', 'v0', {
            'observation_keys': ('observation', ),
            'goal_keys': ('desired_goal', ),
        })
        assert isinstance(env.observation_space, gym.spaces.Dict)

        max_size = 1000
        episode_length = 50

        her_strategy = {
            'type': strategy_type,
            'resampling_probability': resampling_probability,
        }

        pool = create_pool(
            env=env,
            max_size=max_size,
            her_strategy=her_strategy,
        )

        strategy_validator = {
            'random': RandomStrategyValidator,
            'final': FinalStrategyValidator,
            'episode': EpisodeStrategyValidator,
            'future': FutureStrategyValidator,
        }[strategy_type](her_strategy=her_strategy)

        episode_lengths = []
        while pool.size < pool._max_size:
            episode_length = np.random.randint(5, 50)
            episode_lengths.append(episode_length)

            samples = {
                'observations': {
                    name: np.empty(
                        (episode_length, *space.shape), dtype=space.dtype)
                    for name, space in env.observation_space.spaces.items()
                },
                'next_observations': {
                    name: np.empty(
                        (episode_length, *space.shape), dtype=space.dtype)
                    for name, space in env.observation_space.spaces.items()
                },
                'actions': np.empty((episode_length, *env.action_space.shape)),
                'rewards': np.empty((episode_length, 1), dtype=np.float32),
                'terminals': np.empty((episode_length, 1), dtype=bool),
            }

            observation = env.reset()
            for i in range(episode_length):
                action = env.action_space.sample()
                next_observation, reward, terminal, info = env.step(action)
                for name, value in observation.items():
                    samples['observations'][name][i, :] = value
                    samples['next_observations'][name][i, :] = next_observation[name]
                    samples['actions'][i] = action
                    samples['rewards'][i] = reward
                    samples['terminals'][i] = terminal
                observation = next_observation

            pool.add_path(samples)

        for i in range(100):
            random_batch = pool.random_batch(256)
            strategy_validator.verify_batch(random_batch)

        assert strategy_validator.statistics_match()

    @pytest.mark.parametrize("reward_function", REWARD_FUNCTIONS)
    def test_custom_reward_function(self, reward_function):
        return

    @pytest.mark.parametrize("terminal_function", TERMINAL_FUNCTIONS)
    def test_custom_terminal_function(self, terminal_function):
        return


if __name__ == '__main__':
    unittest.main()
