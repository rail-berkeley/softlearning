from dataclasses import dataclass
from typing import Union, Callable
from numbers import Number
import gzip
import pickle

import numpy as np
import tensorflow as tf
import tree

from .replay_pool import ReplayPool


@dataclass
class Field:
    name: str
    dtype: Union[str, np.dtype, tf.DType]
    shape: Union[tuple, tf.TensorShape]
    initializer: Callable = np.zeros
    default_value: Number = 0.0


INDEX_FIELDS = {
    'episode_index_forwards': Field(
        name='episode_index_forwards',
        dtype='uint64',
        shape=(1, ),
        default_value=0,
    ),
    'episode_index_backwards': Field(
        name='episode_index_backwards',
        dtype='uint64',
        shape=(1, ),
        default_value=0,
    ),
}


class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {**fields, **INDEX_FIELDS}
        self.data = tree.map_structure(self._initialize_field, self.fields)

        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0

    @property
    def size(self):
        return self._size

    def _initialize_field(self, field):
        field_shape = (self._max_size, *field.shape)
        field_values = field.initializer(
            field_shape, dtype=field.dtype)

        return field_values

    def _advance(self, count=1):
        """Handles bookkeeping after adding samples to the pool.

        * Moves the pointer (`self._pointer`)
        * Updates the size (`self._size`)
        * Fixes the `episode_index_backwards` field, which might have become
          out of date when the pool is full and we start overriding old
          samples.
        """
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)

        if self.data['episode_index_forwards'][self._pointer] != 0:
            episode_tail_length = int(self.data[
                'episode_index_backwards'
            ][self._pointer, 0] + 1)
            self.data[
                'episode_index_forwards'
            ][np.arange(
                self._pointer, self._pointer + episode_tail_length
            ) % self._max_size] = np.arange(episode_tail_length)[..., None]

        self._samples_since_save += count

    def add_sample(self, sample):
        samples = tree.map_structure(lambda x: x[..., np.newaxis], sample)
        self.add_samples(samples)

    def add_samples(self, samples):
        num_samples = tree.flatten(samples)[0].shape[0]

        assert (('episode_index_forwards' in samples.keys())
                is ('episode_index_backwards' in samples.keys()))
        if 'episode_index_forwards' not in samples.keys():
            samples['episode_index_forwards'] = np.full(
                (num_samples, *self.fields['episode_index_forwards'].shape),
                self.fields['episode_index_forwards'].default_value,
                dtype=self.fields['episode_index_forwards'].dtype)
            samples['episode_index_backwards'] = np.full(
                (num_samples, *self.fields['episode_index_backwards'].shape),
                self.fields['episode_index_backwards'].default_value,
                dtype=self.fields['episode_index_backwards'].dtype)

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        def add_sample(path, data, new_values, field):
            assert new_values.shape[0] == num_samples, (
                new_values.shape, num_samples)
            data[index] = new_values

        tree.map_structure_with_path(
            add_sample, self.data, samples, self.fields)

        self._advance(num_samples)

    def add_path(self, path):
        path = path.copy()
        path_length = tree.flatten(path)[0].shape[0]
        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., np.newaxis],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, np.newaxis],
        })

        return self.add_samples(path)

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def random_sequence_batch(self, batch_size, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.sequence_batch_by_indices(random_indices, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, int(last_n)), self._pointer,
            dtype=int
        ) % self._max_size

        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_sequence_batch(self, last_n, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, int(last_n)), self._pointer,
            dtype=int
        ) % self._max_size

        return self.sequence_batch_by_indices(last_n_indices, **kwargs)

    def filter_fields(self, field_names, field_name_filter):
        if isinstance(field_name_filter, str):
            field_name_filter = [field_name_filter]

        if isinstance(field_name_filter, (list, tuple)):
            field_name_list = field_name_filter

            def filter_fn(field_name):
                return field_name in field_name_list

        else:
            filter_fn = field_name_filter

        filtered_field_names = [
            field_name for field_name in field_names
            if filter_fn(field_name)
        ]

        return filtered_field_names

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         validate_index=True):
        if validate_index and np.any(self.size <= indices % self._max_size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        if field_name_filter is not None:
            raise NotImplementedError("TODO(hartikainen)")

        batch = tree.map_structure(
            lambda field: field[indices % self._max_size], self.data)
        return batch

    def sequence_batch_by_indices(self,
                                  indices,
                                  sequence_length,
                                  field_name_filter=None):
        if np.any(self.size <= indices % self._max_size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")
        if indices.size < 1:
            return self.batch_by_indices(indices)

        sequence_indices = (
            indices[:, None] + np.arange(sequence_length)[None])
        sequence_batch = self.batch_by_indices(
            sequence_indices, validate_index=False)

        if 'mask' in sequence_batch:
            raise ValueError(
                "sequence_batch_by_indices adds a field 'mask' into the batch."
                " There already exists a 'mask' field in the batch. Please"
                " remove it before using sequence_batch. TODO(hartikainen):"
                " Allow mask name to be configured.")

        forward_diffs_0 = np.diff(
            sequence_batch['episode_index_forwards'].astype(np.int64), axis=1)
        forward_diffs_1 = np.pad(
            forward_diffs_0, ([0, 0], [0, 1], [0, 0]),
            mode='constant',
            constant_values=-1)
        cut_and_pad_sample_indices = (
            np.argmax(forward_diffs_1[:, ::1, :] < 1, axis=1)
            + 1)[..., 0]

        sequence_batch['mask'] = np.where(
            np.arange(sequence_length)[None, ...]
            < cut_and_pad_sample_indices[..., None],
            True,
            False)

        return sequence_batch

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_samples = pickle.load(f)

        num_samples = tree.flatten(latest_samples)[0].shape[0]

        def assert_shape(data):
            assert data.shape[0] == num_samples, data.shape

        tree.map_structure(assert_shape, latest_samples)

        self.add_samples(latest_samples)
        self._samples_since_save = 0
