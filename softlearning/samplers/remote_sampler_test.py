import pickle
import unittest
import pytest

from softlearning.environments.utils import get_environment
from softlearning.samplers.remote_sampler import RemoteSampler
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning import policies


@pytest.mark.skip(reason="RemoteSampler is currently broken.")
class RemoteSamplerTest(unittest.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'v3', {})
        self.policy = policies.ContinuousUniformPolicy(
            action_range=(
                self.env.action_space.low,
                self.env.action_space.high,
            ),
            input_shapes=self.env.observation_shape,
            output_shape=self.env.action_shape,
            observation_keys=self.env.observation_keys)
        self.pool = SimpleReplayPool(max_size=100, environment=self.env)
        self.remote_sampler = RemoteSampler(max_path_length=10)

    def test_initialization(self):
        self.assertEqual(self.pool.size, 0)
        self.remote_sampler.initialize(self.env, self.policy, self.pool)
        self.remote_sampler.sample(timeout=10)
        self.assertEqual(self.pool.size, 10)

    def test_serialize_deserialize(self):
        self.assertEqual(self.pool.size, 0)

        self.remote_sampler.initialize(self.env, self.policy, self.pool)

        self.remote_sampler.sample()

        deserialized = pickle.loads(pickle.dumps(self.remote_sampler))
        deserialized.initialize(self.env, self.policy, self.pool)

        self.assertEqual(self.pool.size, 10)

        self.remote_sampler.sample(timeout=10)
        self.assertEqual(self.pool.size, 20)

        deserialized = pickle.loads(pickle.dumps(self.remote_sampler))
        deserialized.initialize(self.env, self.policy, self.pool)

        self.assertTrue(isinstance(
            deserialized.environment, type(self.remote_sampler.environment)))
        self.assertEqual(
            self.remote_sampler._n_episodes, deserialized._n_episodes)
        self.assertEqual(
            self.remote_sampler._max_path_return,
            deserialized._max_path_return)
        self.assertEqual(
            self.remote_sampler._last_path_return,
            deserialized._last_path_return)
        self.assertEqual(
            len(self.remote_sampler._last_n_paths),
            len(deserialized._last_n_paths))

        self.remote_sampler.sample(timeout=10)
        deserialized.sample(timeout=10)

        self.assertEqual(
            self.remote_sampler._n_episodes, deserialized._n_episodes)
        self.assertNotEqual(
            self.remote_sampler._last_path_return,
            deserialized._last_path_return)
        self.assertEqual(
            len(self.remote_sampler._last_n_paths),
            len(deserialized._last_n_paths))


if __name__ == '__main__':
    unittest.main()
