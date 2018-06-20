import numpy as np

from .replay_buffer import ReplayBuffer


class UnionBuffer(ReplayBuffer):
    def __init__(self, buffers):
        buffer_sizes = np.array([b.size for b in buffers])
        self._total_size = sum(buffer_sizes)
        self._normalized_buffer_sizes = buffer_sizes / self._total_size

        self.buffers = buffers

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def terminate_episode(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._total_size

    def add_path(self, **kwargs):
        raise NotImplementedError

    def random_batch(self, batch_size):

        # TODO: Hack
        partial_batch_sizes = self._normalized_buffer_sizes * batch_size
        partial_batch_sizes = partial_batch_sizes.astype(int)
        partial_batch_sizes[0] = batch_size - sum(partial_batch_sizes[1:])

        partial_batches = [
            buffer.random_batch(partial_batch_size) for buffer,
            partial_batch_size in zip(self.buffers, partial_batch_sizes)
        ]

        def all_values(key):
            return [partial_batch[key] for partial_batch in partial_batches]

        keys = partial_batches[0].keys()

        return {key: np.concatenate(all_values(key), axis=0) for key in keys}
