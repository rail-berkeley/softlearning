import abc


class ReplayPool(object):
    """A class used to save and replay data."""

    @abc.abstractmethod
    def add_sample(self, sample):
        """Add a transition tuple."""
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """Clean up pool after episode termination."""
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def add_path(self, path):
        """Add a rollout to the replay pool."""
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """Return a random batch of size `batch_size`."""
        pass
