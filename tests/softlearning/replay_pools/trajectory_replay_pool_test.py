import pickle
import unittest
import numpy as np

from softlearning.replay_pools.trajectory_replay_pool import (
    TrajectoryReplayPool)


def create_pool(max_size=100):
    return TrajectoryReplayPool(
        observation_space=None,
        action_space=None,
        max_size=max_size,
    )


def verify_pools_match(pool1, pool2):
    for key in pool2.__dict__:
        if key == '_trajectories':
            pool1_trajectories = pool1.__dict__[key]
            pool2_trajectories = pool2.__dict__[key]
            for pool1_trajectory, pool2_trajectory in (
                    zip(pool1_trajectories, pool2_trajectories)):
                assert pool1_trajectory.keys() == pool2_trajectory.keys()
                for field_name in pool1_trajectory.keys():
                    np.testing.assert_array_equal(
                        pool1_trajectory[field_name],
                        pool2_trajectory[field_name],
                        f"key '{key}', field_name '{field_name}' doesn't match"
                    )
        else:
            np.testing.assert_array_equal(
                pool1.__dict__[key],
                pool2.__dict__[key],
                f"key '{key}' doesn't match")


class TrajectoryReplayPoolTest(unittest.TestCase):
    def setUp(self):
        self.pool = create_pool(10)

    def test_save_load_latest_experience(self):
        self.assertEqual(self.pool._trajectories_since_save, 0)

        num_trajectories_per_save = self.pool._max_size // 2
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories_per_save)
        ]

        self.pool.add_paths(trajectories)

        self.assertEqual(self.pool.num_trajectories, num_trajectories_per_save)
        self.assertEqual(self.pool.size,
                         num_trajectories_per_save * trajectory_length)
        self.assertEqual(self.pool._trajectories_since_save,
                         num_trajectories_per_save)

        self.pool.save_latest_experience('./tmp/pool_1.pkl')

        self.assertEqual(self.pool._trajectories_since_save, 0)

        self.pool.add_paths(trajectories)

        self.assertEqual(self.pool.size,
                         self.pool._max_size * trajectory_length)
        self.assertEqual(self.pool._trajectories_since_save,
                         num_trajectories_per_save)

        self.pool.save_latest_experience('./tmp/pool_2.pkl')

        self.pool.add_paths(trajectories)

        self.assertEqual(self.pool.size,
                         self.pool._max_size * trajectory_length)
        self.assertEqual(self.pool._trajectories_since_save,
                         num_trajectories_per_save)

        self.pool.save_latest_experience('./tmp/pool_3.pkl')

        pool = create_pool(self.pool._max_size)

        self.assertEqual(pool.size, 0)
        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.num_trajectories, self.pool._max_size // 2)
        self.assertEqual(pool.size,
                         (self.pool._max_size // 2) * trajectory_length)
        pool.load_experience('./tmp/pool_2.pkl')
        self.assertEqual(pool.num_trajectories, self.pool._max_size)
        self.assertEqual(pool.size,
                         (self.pool._max_size) * trajectory_length)
        self.assertEqual(pool.size, self.pool.size)
        pool.load_experience('./tmp/pool_3.pkl')
        self.assertEqual(pool.size, self.pool.size)
        self.assertEqual(pool.size,
                         (self.pool._max_size) * trajectory_length)

        for trajectory1, trajectory2 in zip(
                pool._trajectories, self.pool._trajectories):
            self.assertEqual(trajectory1.keys(), trajectory2.keys())
            for key in trajectory1:
                np.testing.assert_array_equal(trajectory1[key], trajectory2[key])

    def test_save_load_latest_experience_empty_pool(self):
        self.assertEqual(self.pool._trajectories_since_save, 0)
        self.pool.save_latest_experience('./tmp/pool_1.pkl')
        pool = create_pool(self.pool._max_size)
        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.size, 0)

    def test_save_latest_experience_with_overflown_pool(self):
        self.assertEqual(self.pool._trajectories_since_save, 0)

        num_trajectories = self.pool._max_size + 2
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)

        self.assertEqual(self.pool.num_trajectories, self.pool._max_size)
        self.assertEqual(self.pool._trajectories_since_save,
                         self.pool._max_size + 2)
        self.pool.save_latest_experience('./tmp/pool_1.pkl')
        pool = create_pool(self.pool._max_size)
        self.assertEqual(pool.size, 0)

        import gzip
        with gzip.open('./tmp/pool_1.pkl', 'rb') as f:
            latest_trajectories = pickle.load(f)
            self.assertEqual(len(latest_trajectories), self.pool._max_size)

        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.size,
                         self.pool._max_size * trajectory_length)

        for trajectory1, trajectory2 in zip(
                trajectories, self.pool._trajectories):
            self.assertEqual(trajectory1.keys(), trajectory2.keys())
            for field_name in trajectory1:
                np.testing.assert_array_equal(
                    trajectory1[field_name], trajectory2[field_name])

    def test_serialize_deserialize_full(self):
        # Fill fields with random data
        num_trajectories = self.pool._max_size + 2
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)

        self.assertEqual(self.pool.num_trajectories, self.pool._max_size)
        self.assertEqual(self.pool.size,
                         trajectory_length * self.pool._max_size)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)

        verify_pools_match(self.pool, deserialized)
        self.assertNotEqual(id(self.pool), id(deserialized))
        self.assertEqual(deserialized.num_trajectories, self.pool._max_size)
        self.assertEqual(deserialized.size,
                         trajectory_length * self.pool._max_size)

    def test_serialize_deserialize_not_full(self):
        # Fill fields with random data
        num_trajectories = self.pool._max_size - 2
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)
        self.assertEqual(self.pool.num_trajectories, num_trajectories)
        self.assertEqual(self.pool.size,
                         num_trajectories * trajectory_length)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)

        verify_pools_match(self.pool, deserialized)
        self.assertNotEqual(id(self.pool), id(deserialized))
        self.assertEqual(deserialized.num_trajectories, num_trajectories)
        self.assertEqual(deserialized.size,
                         num_trajectories * trajectory_length)

    def test_serialize_deserialize_empty(self):
        # Fill fields with random data

        self.assertEqual(self.pool.num_trajectories, 0)
        self.assertEqual(self.pool.size, 0)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)

        verify_pools_match(self.pool, deserialized)
        self.assertNotEqual(id(self.pool), id(deserialized))
        self.assertEqual(deserialized.num_trajectories, 0)
        self.assertEqual(deserialized.size, 0)

    def test_add_path(self):
        for value in range(self.pool._max_size):
            path = {
                'field1': np.array([value]),
                'field2': np.array([-value*2]),
            }
            self.pool.add_path(path)

        self.assertEqual(len(self.pool._trajectories), self.pool._max_size)

        for i, trajectory in enumerate(self.pool._trajectories):
            np.testing.assert_array_equal(trajectory['field1'], [i])
            np.testing.assert_array_equal(trajectory['field2'], [-i * 2])

    def test_add_paths(self):
        num_trajectories = 4
        path_length = 10
        paths = [
            {
                'field1': np.arange(path_length)[:, None],
                'field2': -np.arange(path_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(paths)

        self.assertEqual(self.pool.num_trajectories, num_trajectories)
        self.assertEqual(self.pool.size, num_trajectories * path_length)

        for trajectory in self.pool._trajectories:
            np.testing.assert_array_equal(
                trajectory['field1'],
                np.arange(path_length)[:, None])
            np.testing.assert_array_equal(
                trajectory['field2'],
                -np.arange(path_length)[:, None] * 2)

    def test_random_batch(self):
        empty_pool_batch = self.pool.random_batch(4)
        self.assertFalse(empty_pool_batch)

        num_trajectories = 4
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)

        full_pool_batch = self.pool.random_batch(4)

        for key, values in full_pool_batch.items():
            self.assertEqual(values.shape, (4, 1))

        self.assertTrue(np.all(full_pool_batch['field1'] >= 0))

        self.assertTrue(np.all(full_pool_batch['field2'] % 2 == 0))
        self.assertTrue(np.all(full_pool_batch['field2'] <= 0))

    def test_random_batch_with_variable_length_trajectories(self):
        batch_size = 256
        num_trajectories = 20
        trajectories = [
            {
                'field1': np.arange(np.random.randint(50, 1000))[:, None],
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)

        batch = self.pool.random_batch(batch_size)
        for key, values in batch.items():
            self.assertEqual(values.shape, (batch_size, 1))

    def test_last_n_batch(self):
        empty_pool_batch = self.pool.last_n_batch(4)
        self.assertFalse(empty_pool_batch)

        num_trajectories = 4
        trajectory_length = 10
        trajectories = [
            {
                'field1': i * np.arange(trajectory_length)[:, None],
                'field2': -i * np.arange(trajectory_length)[:, None] * 2,
            }
            for i in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)
        full_pool_batch = self.pool.last_n_batch(int(trajectory_length * 2.5))

        for key, values in full_pool_batch.items():
            expected = np.concatenate((
                trajectories[-3][key][trajectory_length // 2:],
                trajectories[-2][key],
                trajectories[-1][key]
            ))
            np.testing.assert_array_equal(values, expected)
            self.assertEqual(values.shape, (2.5 * trajectory_length, 1))

        self.pool.add_paths(trajectories)
        full_pool_batch = self.pool.last_n_batch(int(trajectory_length * 2))

        for key, values in full_pool_batch.items():
            expected = np.concatenate((
                trajectories[-2][key],
                trajectories[-1][key]
            ))
            np.testing.assert_array_equal(values, expected)
            self.assertEqual(values.shape, (2 * trajectory_length, 1))

    def test_last_n_batch_with_overflown_pool(self):
        num_trajectories = self.pool._max_size + 2
        trajectory_length = 10
        trajectories = [
            {
                'field1': i * np.arange(trajectory_length)[:, None],
                'field2': -i * np.arange(trajectory_length)[:, None] * 2,
            }
            for i in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)
        full_pool_batch = self.pool.last_n_batch(int(trajectory_length * 2.5))

        for key, values in full_pool_batch.items():
            expected = np.concatenate((
                trajectories[-3][key][trajectory_length // 2:],
                trajectories[-2][key],
                trajectories[-1][key]
            ))
            np.testing.assert_array_equal(values, expected)
            self.assertEqual(values.shape, (2.5 * trajectory_length, 1))

    def test_batch_by_indices(self):
        with self.assertRaises(TypeError):
            self.pool.batch_by_indices(np.array([-1, 2, 4]))

        num_trajectories = 4
        trajectory_length = 10
        trajectories = [
            {
                'field1': np.arange(trajectory_length)[:, None],
                'field2': -np.arange(trajectory_length)[:, None] * 2,
            }
            for _ in range(num_trajectories)
        ]

        self.pool.add_paths(trajectories)

        batch = self.pool.batch_by_indices(
            np.repeat(np.arange(num_trajectories), trajectory_length),
            np.tile(np.flip(np.arange(trajectory_length)), num_trajectories))

        for field_name, values in batch.items():
            field_expected = np.concatenate([
                np.flip(trajectory[field_name]) for trajectory in trajectories])
            np.testing.assert_array_equal(
                batch[field_name],
                field_expected)


if __name__ == '__main__':
    unittest.main()
