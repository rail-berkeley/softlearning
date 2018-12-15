import pickle
import unittest

from softlearning.environments.utils import get_environment
from softlearning.samplers.remote_sampler import RemoteSampler
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
from softlearning.policies.uniform_policy import UniformPolicy


class RemoteSamplerTest(unittest.TestCase):
    def setUp(self):
        self.env = get_environment('gym', 'Swimmer', 'Default', {})
        self.policy = UniformPolicy(
            input_shapes=(self.env.observation_space.shape, ),
            output_shape=self.env.action_space.shape)
        self.pool = SimpleReplayPool(
            max_size=100,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space)
        self.remote_sampler = RemoteSampler(
            max_path_length=10,
            min_pool_size=10,
            batch_size=10)

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
        self.assertEqual(self.pool.size, 10)

        self.remote_sampler.sample(timeout=10)
        self.assertEqual(self.pool.size, 20)

        deserialized = pickle.loads(pickle.dumps(self.remote_sampler))

        self.assertTrue(isinstance(
            deserialized.env, type(self.remote_sampler.env)))
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
