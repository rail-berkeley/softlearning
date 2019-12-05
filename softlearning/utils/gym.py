from gym import spaces


DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box, )


def is_continuous_space(space):
    return isinstance(space, CONTINUOUS_SPACES)


def is_discrete_space(space):
    return isinstance(space, DISCRETE_SPACES)
