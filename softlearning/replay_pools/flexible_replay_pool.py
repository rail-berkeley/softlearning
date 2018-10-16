import numpy as np

from serializable import Serializable
from .replay_pool import ReplayPool


class FlexibleReplayPool(ReplayPool, Serializable):
    def __init__(self, max_size, fields):
        ReplayPool.__init__(self)
        self._Serializable__initialize(locals())

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {}
        self.field_names = []
        self.add_fields(fields)

        self._pointer = 0
        self._size = 0

    @property
    def size(self):
        return self._size

    def add_fields(self, fields):
        self.fields.update(fields)
        self.field_names += list(fields.keys())

        for field_name, field_attrs in fields.items():
            field_shape = (self._max_size, *field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            setattr(self, field_name, initializer(
                field_shape, dtype=field_attrs['dtype']))

    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)

    def add_sample(self, **kwargs):
        self.add_samples(1, **kwargs)

    def add_samples(self, num_samples=1, **kwargs):
        for field_name in self.field_names:
            idx = np.arange(
                self._pointer, self._pointer + num_samples) % self._max_size
            values = (
                kwargs.pop(field_name, None)
                if field_name in kwargs
                else self.fields[field_name]['default_value'])
            getattr(self, field_name)[idx] = values

        assert not kwargs, ("Got unexpected fields in the sample: ", kwargs)

        self._advance(num_samples)

    def __getstate__(self):
        pool_state = super(FlexibleReplayPool, self).__getstate__()
        pool_state.update({
            field_name: getattr(self, field_name).tobytes()
            for field_name in self.field_names
        })

        pool_state.update({
            '_pointer': self._pointer,
            '_size': self._size
        })

        return pool_state

    def __setstate__(self, pool_state):
        super(FlexibleReplayPool, self).__setstate__(pool_state)

        for field_name in self.field_names:
            field = self.fields[field_name]
            flat_values = np.frombuffer(
                pool_state[field_name], dtype=field['dtype'])
            values = flat_values.reshape(
                (self._max_size, *field['shape']))
            setattr(self, field_name, values)

        self._pointer = pool_state['_pointer']
        self._size = pool_state['_size']

    def random_indices(self, batch_size):
        if self._size == 0: return ()
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(random_indices, field_name_filter)

    def batch_by_indices(self, indices, field_name_filter=None):
        field_names = self.field_names
        if field_name_filter is not None:
            field_names = [
                field_name for field_name in field_names
                if field_name_filter(field_name)
            ]

        return {
            field_name: getattr(self, field_name)[indices]
            for field_name in field_names
        }
