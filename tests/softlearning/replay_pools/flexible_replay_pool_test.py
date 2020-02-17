import pickle
import unittest
import numpy as np
import os

import tree

from softlearning.replay_pools.flexible_replay_pool import (
    FlexibleReplayPool, Field, INDEX_FIELDS)


def create_pool(max_size=100, field_shapes=((1,), (1,))):
    return FlexibleReplayPool(
        max_size=max_size,
        fields={
            f'field{i}': Field(
                name=f'field{i}',
                shape=shape,
                dtype='float32')
            for i, shape in enumerate(field_shapes, 1)
        }
    )


class FlexibleReplayPoolTest(unittest.TestCase):
    def setUp(self):
        self.pool = create_pool(100)
        os.makedirs('./tmp', exist_ok=True)

    def test_multi_dimensional_field(self):
        # Fill fields with random data
        pool = create_pool(10, field_shapes=((1, 3), (1, )))
        num_samples = pool._max_size // 2
        pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in pool.fields.items()
        })

        self.assertEqual(pool._size, num_samples)

        serialized = pickle.dumps(pool)
        deserialized = pickle.loads(serialized)
        for key in deserialized.__dict__:
            if key == 'data':
                for field_name in pool.__dict__[key]:
                    np.testing.assert_array_equal(
                        pool.__dict__[key][field_name],
                        deserialized.__dict__[key][field_name])
            else:
                np.testing.assert_array_equal(
                    pool.__dict__[key],
                    deserialized.__dict__[key])

        self.assertNotEqual(id(pool), id(deserialized))

        self.assertEqual(deserialized._size, num_samples)
        for field_name, field_attrs in pool.fields.items():
            np.testing.assert_array_equal(
                pool.fields[field_name],
                deserialized.fields[field_name])

    def test_advance(self):
        # Fill fields with random data
        pool = create_pool(10, field_shapes=((1, 3), (1, )))
        num_samples = pool._max_size - 2
        pool.add_path({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in pool.fields.items()
        })

        self.assertEqual(pool._size, num_samples)
        self.assertEqual(pool._pointer, num_samples)

        np.testing.assert_equal(
            pool.data['episode_index_forwards'],
            np.concatenate((
                np.arange(num_samples)[..., None],
                np.zeros((pool._max_size-num_samples, 1)),
            ), axis=0))

        np.testing.assert_equal(
            pool.data['episode_index_backwards'],
            np.concatenate((
                np.arange(num_samples-1, -1, -1)[..., None],
                np.zeros((pool._max_size-num_samples, 1)),
            ), axis=0))

        pool.add_path({
            field_name: np.random.uniform(
                0, 1, (pool._max_size - 2, *field_attrs.shape))
            for field_name, field_attrs in pool.fields.items()
        })

        self.assertEqual(pool._size, pool._max_size)
        self.assertEqual(pool._pointer, num_samples - 2)

        np.testing.assert_equal(
            pool.data['episode_index_forwards'],
            np.concatenate((
                np.arange(2, pool._max_size-2)[..., None],
                np.arange(2)[..., None],
                np.arange(2)[..., None],
            ), axis=0))

        np.testing.assert_equal(
            pool.data['episode_index_backwards'],
            np.concatenate((
                np.arange(5, -1, -1)[..., None],
                np.arange(1, -1, -1)[..., None],
                np.arange(7, 5, -1)[..., None],
            ), axis=0))

        # Make sure that overflowing episode index gets correctly fixed.
        pool.add_path({
            field_name: np.random.uniform(
                0, 1, (3, *field_attrs.shape))
            for field_name, field_attrs in pool.fields.items()
        })

        self.assertEqual(pool._size, pool._max_size)
        self.assertEqual(pool._pointer, num_samples+1)

        np.testing.assert_equal(
            pool.data['episode_index_forwards'],
            np.concatenate((
                np.arange(1, num_samples-1)[..., None],
                np.arange(3)[..., None],
                np.arange(1)[..., None],
            ), axis=0))

        np.testing.assert_equal(
            pool.data['episode_index_backwards'],
            np.concatenate((
                np.arange(5, -1, -1)[..., None],
                np.arange(2, -1, -1)[..., None],
                np.arange(6, 5, -1)[..., None],
            ), axis=0))

    def test_save_load_latest_experience(self):
        self.assertEqual(self.pool._samples_since_save, 0)

        num_samples = self.pool._max_size // 2
        self.pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in self.pool.fields.items()
        })

        self.assertEqual(self.pool.size, self.pool._max_size // 2)
        self.assertEqual(self.pool._samples_since_save, self.pool.size)

        self.pool.save_latest_experience('./tmp/pool_1.pkl')

        self.assertEqual(self.pool._samples_since_save, 0)

        self.pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in self.pool.fields.items()
        })

        self.assertEqual(self.pool.size, self.pool._max_size)
        self.assertEqual(
            self.pool._samples_since_save, self.pool._max_size // 2)

        self.pool.save_latest_experience('./tmp/pool_2.pkl')

        self.pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in self.pool.fields.items()
        })

        self.assertEqual(self.pool.size, self.pool._max_size)
        self.assertEqual(
            self.pool._samples_since_save, self.pool._max_size // 2)

        self.pool.save_latest_experience('./tmp/pool_3.pkl')

        pool = create_pool(self.pool._max_size)

        self.assertEqual(pool.size, 0)
        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.size, self.pool._max_size // 2)
        pool.load_experience('./tmp/pool_2.pkl')
        self.assertEqual(pool.size, self.pool.size)
        pool.load_experience('./tmp/pool_3.pkl')
        self.assertEqual(pool.size, self.pool.size)

        for field_name, field_attrs in pool.fields.items():
            np.testing.assert_array_equal(
                pool.fields[field_name],
                self.pool.fields[field_name])

    def test_save_load_latest_experience_empty_pool(self):
        self.assertEqual(self.pool._samples_since_save, 0)
        self.pool.save_latest_experience('./tmp/pool_1.pkl')
        pool = create_pool(self.pool._max_size)
        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.size, 0)

    def test_save_latest_experience_with_overflown_pool(self):
        self.assertEqual(self.pool._samples_since_save, 0)

        num_samples = self.pool._max_size + 10
        samples = {
            'field1': np.arange(num_samples)[:, None],
            'field2': -np.arange(num_samples)[:, None] * 2,
        }
        self.pool.add_samples(samples)

        self.assertEqual(self.pool.size, self.pool._max_size)
        self.assertEqual(self.pool._samples_since_save, num_samples)
        self.pool.save_latest_experience('./tmp/pool_1.pkl')
        pool = create_pool(self.pool._max_size)
        self.assertEqual(pool.size, 0)

        import gzip
        with gzip.open('./tmp/pool_1.pkl', 'rb') as f:
            latest_samples = pickle.load(f)

            def assert_same_shape(field, data):
                expected_shape = (self.pool._max_size, *field.shape)
                self.assertEqual(data.shape, expected_shape)

            tree.map_structure(
                assert_same_shape, self.pool.fields, latest_samples)

        pool.load_experience('./tmp/pool_1.pkl')
        self.assertEqual(pool.size, self.pool._max_size)

        assert all(
            index_field in pool.fields.keys()
            for index_field in INDEX_FIELDS)

        def assert_field_data_shape(field_data, field_samples):
            np.testing.assert_array_equal(
                field_data, field_samples[-self.pool._max_size:])

        tree.map_structure(assert_field_data_shape, pool.data, samples)

    def test_field_initialization(self):
        def verify_field(field_attrs, field_values):
            self.assertEqual(field_values.shape,
                             (self.pool._max_size, *field_attrs.shape))
            self.assertEqual(field_values.dtype.name, field_attrs.dtype)

            np.testing.assert_array_equal(field_values, 0.0)

        tree.map_structure(verify_field, self.pool.fields, self.pool.data)

    def test_serialize_deserialize_full(self):
        # Fill fields with random data
        self.pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (self.pool._max_size, *field_attrs.shape))
            for field_name, field_attrs in self.pool.fields.items()
        })

        self.assertEqual(self.pool._size, self.pool._max_size)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)

        for key in deserialized.__dict__:
            if key == 'data':
                for field_name in self.pool.__dict__[key]:
                    np.testing.assert_array_equal(
                        self.pool.__dict__[key][field_name],
                        deserialized.__dict__[key][field_name])
            else:
                np.testing.assert_array_equal(
                    self.pool.__dict__[key],
                    deserialized.__dict__[key])

        self.assertNotEqual(id(self.pool), id(deserialized))

        self.assertEqual(deserialized._size, deserialized._max_size)
        for field_name, field_attrs in self.pool.fields.items():
            np.testing.assert_array_equal(
                self.pool.fields[field_name],
                deserialized.fields[field_name])

    def test_serialize_deserialize_not_full(self):
        # Fill fields with random data
        num_samples = self.pool._max_size // 2
        self.pool.add_samples({
            field_name: np.random.uniform(
                0, 1, (num_samples, *field_attrs.shape))
            for field_name, field_attrs in self.pool.fields.items()
        })

        self.assertEqual(self.pool._size, num_samples)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)
        for key in deserialized.__dict__:
            if key == 'data':
                for field_name in self.pool.__dict__[key]:
                    np.testing.assert_array_equal(
                        self.pool.__dict__[key][field_name],
                        deserialized.__dict__[key][field_name])
            else:
                np.testing.assert_array_equal(
                    self.pool.__dict__[key],
                    deserialized.__dict__[key])

        self.assertNotEqual(id(self.pool), id(deserialized))

        self.assertEqual(deserialized._size, num_samples)
        for field_name, field_attrs in self.pool.fields.items():
            np.testing.assert_array_equal(
                self.pool.fields[field_name],
                deserialized.fields[field_name])

    def test_serialize_deserialize_empty(self):
        self.assertEqual(self.pool._size, 0)
        tree.map_structure(
            lambda field_data: np.testing.assert_array_equal(field_data, 0.0),
            self.pool.data)

        serialized = pickle.dumps(self.pool)
        deserialized = pickle.loads(serialized)
        for key in deserialized.__dict__:
            if key == 'data':
                for field_name in self.pool.__dict__[key]:
                    np.testing.assert_array_equal(
                        self.pool.__dict__[key][field_name],
                        deserialized.__dict__[key][field_name])
            else:
                np.testing.assert_array_equal(
                    self.pool.__dict__[key],
                    deserialized.__dict__[key])

        self.assertNotEqual(id(self.pool), id(deserialized))

        self.assertEqual(deserialized._size, 0)
        for field_name, field_attrs in self.pool.fields.items():
            np.testing.assert_array_equal(
                 self.pool.fields[field_name],
                 deserialized.fields[field_name])

    def test_add_sample(self):
        for value in range(self.pool._max_size):
            sample = {
                'field1': np.array([value]),
                'field2': np.array([-value*2]),
            }
            self.pool.add_sample(sample)

        np.testing.assert_array_equal(
            self.pool.data['field1'],
            np.arange(self.pool._max_size)[:, None])
        np.testing.assert_array_equal(
            self.pool.data['field2'],
            -np.arange(self.pool._max_size)[:, None] * 2)

    def test_add_samples(self):
        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(samples)

        np.testing.assert_array_equal(
            self.pool.data['field1'],
            np.arange(self.pool._max_size)[:, None])
        np.testing.assert_array_equal(
            self.pool.data['field2'],
            -np.arange(self.pool._max_size)[:, None] * 2)

    def test_random_indices(self):
        empty_pool_indices = self.pool.random_indices(4)
        self.assertEqual(empty_pool_indices.shape, (0, ))
        self.assertEqual(empty_pool_indices.dtype, np.int64)

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(samples)
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
        self.pool.add_samples(samples)
        full_pool_batch = self.pool.random_batch(4)

        for key, values in full_pool_batch.items():
            self.assertEqual(values.shape, (4, 1))

        self.assertTrue(
            np.all(full_pool_batch['field1'] < self.pool._max_size))
        self.assertTrue(np.all(full_pool_batch['field1'] >= 0))

        self.assertTrue(np.all(full_pool_batch['field2'] % 2 == 0))
        self.assertTrue(np.all(full_pool_batch['field2'] <= 0))

    def test_random_sequence_batch(self):
        empty_pool_batch = self.pool.random_sequence_batch(4, sequence_length=10)
        for key, values in empty_pool_batch.items():
            self.assertEqual(values.size, 0)

        sequence_length = 10

        path_lengths = [10, 4, 50, 36]
        assert sum(path_lengths) == self.pool._max_size, path_lengths

        for path_length in path_lengths:
            samples = {
                'field1': np.arange(path_length)[:, None],
                'field2': -np.arange(path_length)[:, None] * 2,
            }
            self.pool.add_path(samples)
        full_pool_batch = self.pool.random_sequence_batch(
            4, sequence_length=sequence_length)

        for key, values in full_pool_batch.items():
            self.assertEqual(values.shape, (4, sequence_length, 1))

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
        self.pool.add_samples(samples)
        full_pool_batch = self.pool.last_n_batch(4)

        assert all(
            index_field in full_pool_batch.keys()
            for index_field in INDEX_FIELDS)
        for key, values in full_pool_batch.items():
            if key in INDEX_FIELDS: continue
            np.testing.assert_array_equal(samples[key][-4:], values)
            self.assertEqual(values.shape, (4, 1))

    def test_last_n_sequence_batch(self):
        empty_pool_batch = self.pool.last_n_sequence_batch(
            4, sequence_length=10)
        for key, values in empty_pool_batch.items():
            self.assertEqual(values.size, 0)

        sequence_length = 2

        path_lengths = [10, 4, 50, 36]
        assert sum(path_lengths) == self.pool._max_size, path_lengths

        for path_length in path_lengths:
            samples = {
                'field1': np.arange(path_length)[:, None],
                'field2': -np.arange(path_length)[:, None] * 2,
            }
            self.pool.add_path(samples)
        full_pool_batch = self.pool.last_n_sequence_batch(
            4, sequence_length=sequence_length)

        for key, values in full_pool_batch.items():
            self.assertEqual(values.shape, (4, sequence_length, 1))

        assert all(
            index_field in full_pool_batch.keys()
            for index_field in INDEX_FIELDS)
        for key, values in full_pool_batch.items():
            if key in INDEX_FIELDS: continue
            np.testing.assert_array_equal(
                samples[key][
                    np.stack((np.arange(-5, -1), np.arange(-4, 0))).T
                ],
                values)
            self.assertEqual(values.shape, (4, sequence_length, 1))

    def test_last_n_batch_with_overflown_pool(self):
        samples = {
            'field1': np.arange(self.pool._max_size + 10)[:, None],
            'field2': -np.arange(self.pool._max_size + 10)[:, None] * 2,
        }
        self.pool.add_samples(samples)
        full_pool_batch = self.pool.last_n_batch(20)

        assert all(
            index_field in full_pool_batch.keys()
            for index_field in INDEX_FIELDS)
        for key, values in full_pool_batch.items():
            if key in INDEX_FIELDS: continue
            np.testing.assert_array_equal(
                samples[key][-20:], values)
            self.assertEqual(values.shape, (20, 1))

    def test_batch_by_indices(self):
        with self.assertRaises(ValueError):
            self.pool.batch_by_indices(np.array([-1, 2, 4]))

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(samples)

        batch = self.pool.batch_by_indices(
            np.flip(np.arange(self.pool._max_size)))
        assert all(
            index_field in batch.keys() for index_field in INDEX_FIELDS)
        for key, values in batch.items():
            if key in INDEX_FIELDS: continue
            np.testing.assert_array_equal(np.flip(samples[key]), values)
            self.assertEqual(values.shape, (self.pool._max_size, 1))

    def test_sequence_overlaps_two_episodes(self):
        sequence_length = self.pool._max_size
        path_lengths = [
            self.pool._max_size // 2 - 5,
            (self.pool._max_size // 2) - 3,
            8,
        ]

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[::-1, None] * 2,
        }
        for path_end, path_length in zip(
                np.cumsum(path_lengths), path_lengths):
            self.pool.add_path(tree.map_structure(
                lambda s: s[path_end - path_length:path_end],
                samples))

        batch_indices = np.array([0, *np.cumsum(path_lengths), -2, -1])
        batch = self.pool.sequence_batch_by_indices(
            batch_indices, sequence_length=sequence_length)

        for key, values in batch.items():
            self.assertEqual(
                values.shape, (batch_indices.size, sequence_length, 1))

        self.assertTrue(all(index_field in batch.keys()
                            for index_field in INDEX_FIELDS))

        for i, (episode_start_index, episode_length) in enumerate(zip(
                (0, *np.cumsum(path_lengths)[:-1]), path_lengths)):
            np.testing.assert_equal(batch['field1'][i][:-1], 0)
            np.testing.assert_equal(
                batch['field1'][i][-1],
                samples['field1'][episode_start_index])
            np.testing.assert_equal(batch['field2'][i][:-1], 0)
            np.testing.assert_equal(
                batch['field2'][i][-1],
                samples['field2'][episode_start_index])

            np.testing.assert_equal(batch['episode_index_forwards'][i], 0)
            np.testing.assert_equal(
                batch['episode_index_backwards'][i][:-1], 0)
            np.testing.assert_equal(
                batch['episode_index_backwards'][i][-1], episode_length - 1)

        np.testing.assert_equal(
            batch['field1'][-2][:- path_lengths[-1] + 1], 0)
        np.testing.assert_equal(
            batch['field1'][-2][- path_lengths[-1] + 1:],
            samples['field1'][-path_lengths[-1]:-1])

        np.testing.assert_equal(
            batch['field2'][-2][:- path_lengths[-1] + 1], 0)
        np.testing.assert_equal(
            batch['field2'][-2][- path_lengths[-1] + 1:],
            samples['field2'][-path_lengths[-1]:-1])

        np.testing.assert_equal(
            batch['field1'][-1][:- path_lengths[-1]], 0)
        np.testing.assert_equal(
            batch['field1'][-1][- path_lengths[-1]:],
            samples['field1'][-path_lengths[-1]:])

        np.testing.assert_equal(
            batch['field2'][-1][:- path_lengths[-1]], 0)
        np.testing.assert_equal(
            batch['field2'][-1][- path_lengths[-1]:],
            samples['field2'][-path_lengths[-1]:])

    def test_sequence_batch_by_indices(self):
        sequence_length = 2

        with self.assertRaises(ValueError):
            self.pool.sequence_batch_by_indices(
                np.array([-1, 2, 4]), sequence_length=sequence_length)

        path_lengths = [10, 4, 50, 36]
        assert sum(path_lengths) == self.pool._max_size, path_lengths

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        for path_end, path_length in zip(
                np.cumsum(path_lengths), path_lengths):
            self.pool.add_path(tree.map_structure(
                lambda s: s[path_end - path_length:path_end],
                samples))
        batch_indices = np.flip(np.arange(self.pool._max_size))
        full_pool_batch = self.pool.sequence_batch_by_indices(
            batch_indices, sequence_length=sequence_length)

        for key, values in full_pool_batch.items():
            self.assertEqual(
                values.shape, (self.pool._max_size, sequence_length, 1))

        self.assertTrue(all(index_field in full_pool_batch.keys()
                            for index_field in INDEX_FIELDS))

        episode_start_indices = np.flatnonzero(
            self.pool.data['episode_index_forwards'] == 0)
        episode_start_indices_batch = np.flatnonzero(
            np.isin(batch_indices, episode_start_indices))

        for key, values in full_pool_batch.items():
            if key in INDEX_FIELDS: continue
            expected = np.stack((
                np.roll(np.flip(samples[key]), -1),
                np.flip(samples[key]),
            ), axis=1)
            expected[episode_start_indices_batch, 0, :] = 0
            np.testing.assert_array_equal(expected, values)
            self.assertEqual(
                values.shape,
                (self.pool._max_size, sequence_length, 1))

            # Make sure that the values at the start of the episode are zero.
            np.testing.assert_equal(
                values[episode_start_indices_batch][:, :-1, :], 0)
            values[episode_start_indices_batch]

    def test_batch_by_indices_with_filter(self):
        with self.assertRaises(ValueError):
            self.pool.batch_by_indices(np.array([-1, 2, 4]))

        samples = {
            'field1': np.arange(self.pool._max_size)[:, None],
            'field2': -np.arange(self.pool._max_size)[:, None] * 2,
        }
        self.pool.add_samples(samples)

        with self.assertRaises(NotImplementedError):
           batch = self.pool.batch_by_indices(
               np.flip(np.arange(self.pool._max_size)),
               field_name_filter=('field1', ))


if __name__ == '__main__':
    unittest.main()
