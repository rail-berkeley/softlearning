from dataclasses import dataclass
from typing import Union, Callable
from numbers import Number
import gzip
import pickle

import numpy as np
import tensorflow as tf

from flatten_dict import flatten, unflatten
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
    ),
    'episode_index_backwards': Field(
        name='episode_index_backwards',
        dtype='uint64',
        shape=(1, ),
    ),
}


class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.data = {}
        self.fields = {**fields, **INDEX_FIELDS}
        self.fields_flat = flatten(self.fields)
        self._initialize_data()

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

    def _initialize_data(self):
        """Initialize data for the pool."""
        fields = flatten(self.fields)
        for field_name, field_attrs in fields.items():
            self.data[field_name] = self._initialize_field(field_attrs)

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

        if self.data[('episode_index_forwards', )][self._pointer] != 0:
            episode_tail_length = int(self.data[
                ('episode_index_backwards', )
            ][self._pointer, 0] + 1)
            self.data[
                ('episode_index_forwards', )
            ][np.arange(
                self._pointer, self._pointer + episode_tail_length
            ) % self._max_size] = np.arange(episode_tail_length)[..., None]

        self._samples_since_save += count

    def add_sample(self, sample):
        sample_flat = flatten(sample)
        samples_flat = type(sample)([
            (field_name_flat, np.array(sample_flat[field_name_flat])[None, ...])
            for field_name_flat in sample_flat.keys()
        ])
        samples = unflatten(samples_flat)

        self.add_samples(samples)

    def add_samples(self, samples):
        samples = flatten(samples)

        field_names = tuple(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        for field_name, values in samples.items():
            default_value = self.fields_flat[field_name].default_value
            values = samples.get(field_name, default_value)
            assert values.shape[0] == num_samples
            self.data[field_name][index] = values

        self._advance(num_samples)

    def add_path(self, path):
        path = path.copy()

        path_flat = flatten(path)
        path_length = path_flat[next(iter(path_flat.keys()))].shape[0]
        path.update({
            'episode_index_forwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_forwards'].dtype
            )[..., None],
            'episode_index_backwards': np.arange(
                path_length,
                dtype=self.fields['episode_index_backwards'].dtype
            )[::-1, None],
        })

        return self.add_samples(path)

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, int(last_n)), self._pointer,
            dtype=int
        ) % self._max_size

        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

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

    def batch_by_indices(self, indices, field_name_filter=None):
        if np.any(indices % self._max_size > self.size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        field_names_flat = self.fields_flat.keys()
        if field_name_filter is not None:
            field_names_flat = self.filter_fields(
                field_names_flat, field_name_filter)

        batch_flat = {
            field_name: self.data[field_name][indices]
            for field_name in field_names_flat
        }

        batch = unflatten(batch_flat)
        return batch

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_samples = pickle.load(f)

        latest_samples_flat = flatten(latest_samples)

        key = list(latest_samples_flat.keys())[0]
        num_samples = latest_samples_flat[key].shape[0]
        for data in latest_samples_flat.values():
            assert data.shape[0] == num_samples, data.shape

        self.add_samples(latest_samples)
        self._samples_since_save = 0

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['fields'] = {
    #         field_name: self.data[field_name][:self.size]
    #         for field_name in self.field_names
    #     }

    #     return state

    # def __setstate__(self, state):
    #     if state['_size'] < state['_max_size']:
    #         pad_size = state['_max_size'] - state['_size']
    #         for field_name in state['data'].keys():
    #             field_shape = state['fields'][field_name]['shape']
    #             state['fields'][field_name] = np.concatenate((
    #                 state['fields'][field_name],
    #                 np.zeros((pad_size, *field_shape))
    #             ), axis=0)

    #     self.__dict__ = state
