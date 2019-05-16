from collections import defaultdict
import pickle
import unittest
import numpy as np
import gym

from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.replay_pools.flexible_replay_pool import Field
from softlearning.environments.utils import get_environment


def create_pool(env, max_size=100):
    return SimpleReplayPool(
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_size=max_size,
    )


class SimpleReplayPoolTest(unittest.TestCase):
    def test_create_pool(self):
        env = get_environment('gym', 'Swimmer', 'v3', {})
        pool = create_pool(env=env, max_size=100)

        def verify_field(field, expected_name, expected_dtype, expected_shape):
            self.assertIsInstance(field, Field)
            self.assertEqual(field.name, expected_name)
            self.assertEqual(field.dtype, expected_dtype)
            self.assertEqual(field.shape, expected_shape)
            self.assertEqual(field.initializer, np.zeros)
            self.assertEqual(field.default_value, 0.0)

        verify_field(
            pool.fields['observations'],
            'observations',
            env.observation_space.dtype,
            env.observation_space.shape)

        verify_field(
            pool.fields['next_observations'],
            'next_observations',
            env.observation_space.dtype,
            env.observation_space.shape)

        verify_field(
            pool.fields['actions'],
            'actions',
            env.action_space.dtype,
            env.action_space.shape)

        verify_field(pool.fields['rewards'], 'rewards', 'float32', (1, ))
        verify_field(pool.fields['terminals'], 'terminals', 'bool', (1, ))

    def test_add_samples(self):
        env = get_environment('gym', 'Swimmer', 'v3', {})
        pool = create_pool(env=env, max_size=100)

        observation = env.reset()

        num_samples = pool._max_size // 2

        samples = {
            'observations': np.empty(
                (num_samples, *env.observation_space.shape),
                dtype=env.observation_space.dtype),
            'next_observations': np.empty(
                (num_samples, *env.observation_space.shape),
                dtype=env.observation_space.dtype),
            'actions': np.empty(
                (num_samples, *env.action_space.shape),
                dtype=env.action_space.dtype),
            'rewards': np.empty((num_samples, 1), dtype=np.float32),
            'terminals': np.empty((num_samples, 1), dtype=bool),
        }

        for i in range(num_samples):
            action = env.action_space.sample()
            next_observation, reward, terminal, info = env.step(action)
            # for name, value in observation.items():
            samples['observations'][i] = observation
            samples['next_observations'][i] = next_observation
            samples['actions'][i] = action
            samples['rewards'][i] = reward
            samples['terminals'][i] = terminal

            observation = next_observation

        pool.add_samples(samples)
        last_n_batch = pool.last_n_batch(num_samples)
        np.testing.assert_equal(last_n_batch, samples)


if __name__ == '__main__':
    unittest.main()
