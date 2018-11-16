import pickle
import unittest
import numpy as np

from softlearning.replay_pools.flexible_replay_pool import FlexibleReplayPool


class FlexibleReplayPoolTest(unittest.TestCase):
    def setUp(self):
        self.pool = FlexibleReplayPool(
            max_size=10,
            fields={
                'field1': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                'field2': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
            }
        )

    def test_field_initialization(self):
        # Fill fields with random data
        for field_name, field_attrs in self.pool.fields.items():
            field_values = getattr(self.pool, field_name)
            self.assertEqual(field_values.shape,
                             (self.pool._max_size, *field_attrs['shape']))
            self.assertEqual(field_values.dtype.name, field_attrs['dtype'])

            np.testing.assert_array_equal(field_values, 0.0)

    def test_serialize_deserialize(self):
        # Fill fields with random data
        for field_name, field_attrs in self.pool.fields.items():
            setattr(self.pool,
                    field_name,
                    np.random.uniform(0, 1, (
                        self.pool._max_size, *field_attrs['shape'])))

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)
        for key in deserialized.__dict__:
            np.testing.assert_array_equal(
                self.pool.__dict__[key], deserialized.__dict__[key])

        self.assertNotEqual(id(self.pool), id(deserialized))

        for field_name, field_attrs in self.pool.fields.items():
            np.testing.assert_array_equal(
                getattr(self.pool, field_name),
                getattr(deserialized, field_name))

    def test_add_sample(self):
        for value in range(self.pool._max_size):
            sample = {
                'field1': np.array([[value]]),
                'field2': np.array([[-value*2]]),
            }
            self.pool.add_sample(**sample)

        np.testing.assert_array_equal(
            self.pool.field1, np.arange(self.pool._max_size)[:, None])
        np.testing.assert_array_equal(
            self.pool.field2, -np.arange(self.pool._max_size)[:, None] * 2)

    def test_add_samples(self):
        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(num_samples=self.pool._max_size, **samples)

        np.testing.assert_array_equal(
            self.pool.field1, np.arange(self.pool._max_size)[:, None])
        np.testing.assert_array_equal(
            self.pool.field2, -np.arange(self.pool._max_size)[:, None] * 2)

    def test_random_indices(self):
        empty_pool_indices = self.pool.random_indices(4)
        self.assertEqual(empty_pool_indices.shape, (0, ))
        self.assertEqual(empty_pool_indices.dtype, np.int64)

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(num_samples=self.pool._max_size, **samples)
        full_pool_indices = self.pool.random_indices(4)
        self.assertEqual(full_pool_indices.shape, (4, ))
        self.assertTrue(np.all(full_pool_indices < self.pool.size))
        self.assertTrue(np.all(full_pool_indices >= 0))

    def test_random_batch(self):
        empty_pool_batch = self.pool.random_batch(4)
        for key, values in empty_pool_batch.items():
            self.assertEqual(values.size, 0)

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(num_samples=self.pool._max_size, **samples)
        full_pool_batch = self.pool.random_batch(4)

        for key, values in full_pool_batch.items():
            self.assertEqual(values.shape, (4, 1))

        self.assertTrue(
            np.all(full_pool_batch['field1'] < self.pool._max_size))
        self.assertTrue(np.all(full_pool_batch['field1'] >= 0))

        self.assertTrue(np.all(full_pool_batch['field2'] % 2 == 0))
        self.assertTrue(np.all(full_pool_batch['field2'] <= 0))

    def test_last_n_batch(self):
        empty_pool_batch = self.pool.last_n_batch(4)
        for key, values in empty_pool_batch.items():
            self.assertEqual(values.size, 0)

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(num_samples=self.pool._max_size, **samples)
        full_pool_batch = self.pool.last_n_batch(4)

        for key, values in full_pool_batch.items():
            np.testing.assert_array_equal(samples[key][-4:], values)
            self.assertEqual(values.shape, (4, 1))

    def test_batch_by_indices(self):
        with self.assertRaises(ValueError):
            self.pool.batch_by_indices(np.array([-1, 2, 4]))

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(num_samples=self.pool._max_size, **samples)

        batch = self.pool.batch_by_indices(
            np.flip(np.arange(self.pool._max_size)))
        for key, values in batch.items():
            np.testing.assert_array_equal(np.flip(samples[key]), values)
            self.assertEqual(values.shape, (self.pool._max_size, 1))
        pass


if __name__ == '__main__':
    unittest.main()
