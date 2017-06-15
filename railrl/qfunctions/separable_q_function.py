import abc


class SeparableQFunction(metaclass=abc.ABCMeta):
    """
    A Q-function that's split up into

    Q(state, action) = A(state, action) + V(state)
    """
    @abc.abstractproperty
    def value_function(self):
        pass

    @abc.abstractproperty
    def advantage_function(self):
        pass