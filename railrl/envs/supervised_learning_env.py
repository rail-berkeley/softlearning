import abc


class SupervisedLearningEnv(metaclass=abc.ABCMeta):
    """
    An environment that's really just a supervised learning task.
    """

    @abc.abstractmethod
    def get_batch(self, batch_size):
        """

        :param batch_size: Size of the batch size
        :return: tuple (X, Y) where
            X is a numpy array of size (
                batch_size, self.sequence_length, self.feature_dim
            )
            Y is a numpy array of size (
                batch_size, self.sequence_length, self.target_dim
            )
        """
        pass

    @property
    @abc.abstractmethod
    def feature_dim(self):
        """
        :return: Integer. Dimension of the features.
        """
        pass

    @property
    @abc.abstractmethod
    def target_dim(self):
        """
        :return: Integer. Dimension of the target.
        """
        pass

    @property
    @abc.abstractmethod
    def sequence_length(self):
        """
        :return: Integer. Dimension of the target.
        """
        pass
