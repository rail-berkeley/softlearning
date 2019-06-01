import unittest
import numpy as np
import gym

from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.replay_pools.flexible_replay_pool import Field
from softlearning.environments.utils import get_environment


def create_pool(env, max_size=100):
    return SimpleReplayPool(environment=env, max_size=max_size)


class SimpleReplayPoolTest(unittest.TestCase):
    def test_create_pool(self):
        env = get_environment('gym', 'Swimmer', 'v3', {})
        self.assertIsInstance(env.observation_space, gym.spaces.Dict)
        pool = create_pool(env=env, max_size=100)

        def verify_field(field, expected_name, expected_dtype, expected_shape):
            self.assertIsInstance(field, Field)
            self.assertEqual(field.name, expected_name)
            self.assertEqual(field.dtype, expected_dtype)
            self.assertEqual(field.shape, expected_shape)
            self.assertEqual(field.initializer, np.zeros)
            self.assertEqual(field.default_value, 0.0)

        self.assertIsInstance(pool.fields['observations'], dict)
        for name, space in env.observation_space.spaces.items():
            self.assertIn(name, pool.fields)
            field = pool.fields['observations'][name]
            verify_field(field, name, space.dtype, space.shape)

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

        env.reset()

        num_samples = pool._max_size // 2

        samples = {
            'observations': {
                name: np.empty((num_samples, *space.shape), dtype=space.dtype)
                for name, space in env.observation_space.spaces.items()
            },
            'next_observations': {
                name: np.empty((num_samples, *space.shape), dtype=space.dtype)
                for name, space in env.observation_space.spaces.items()
            },
            'actions': np.empty((num_samples, *env.action_space.shape)),
            'rewards': np.empty((num_samples, 1), dtype=np.float32),
            'terminals': np.empty((num_samples, 1), dtype=bool),
        }

        for i in range(num_samples):
            action = env.action_space.sample()
            observation, reward, terminal, info = env.step(action)
            for name, value in observation.items():
                samples['observations'][name][i, :] = value
                samples['next_observations'][name][i, :] = value
            samples['actions'][i] = action
            samples['rewards'][i] = reward
            samples['terminals'][i] = terminal

        pool.add_path(samples)
        last_n_batch = pool.last_n_batch(num_samples)
        np.testing.assert_equal(
            {
                key: value
                for key, value in last_n_batch.items()
                if key not in
                ('episode_index_backwards', 'episode_index_forwards')
            },
            samples)


if __name__ == '__main__':
    unittest.main()
