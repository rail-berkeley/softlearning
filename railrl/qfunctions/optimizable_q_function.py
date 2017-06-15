import abc


class OptimizableQFunction(metaclass=abc.ABCMeta):
    """
    A Q-function that implicitly has a _policy.
    """
    @abc.abstractproperty
    def implicit_policy(self):
        pass
