import abc


class ReplayPool(object):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay pool know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay pool.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by railrl.samplers.util.rollout
        """
        path_length = path['observations'].shape[0]
        self.add_samples(num_samples=path_length, **{
            key: value
            for key, value in path.items()
            if key in self.field_names
        })
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass
