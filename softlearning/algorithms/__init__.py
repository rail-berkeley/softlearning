from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)

from .sql import SQL  # noqa: unused-import
from .sac import SAC  # noqa: unused-import


def serialize(algorithm):
    return serialize_softlearning_object(algorithm)


def deserialize(name, custom_objects=None):
    """Returns a algorithm function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Algorithm function or class denoted by input string.

    For example:
    >>> softlearning.algorithms.get({'class_name': 'SAC', ...})
      <softlearning.algorithms.sac.SAC object at 0x7fea93d6cdd0>
    >>> softlearning.algorithms.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown algorithm: abcd

    Args:
      name: The name of the algorithm.

    Raises:
        ValueError: `Unknown algorithm` if the input string does not
        denote any defined algorithm.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='algorithm')


def get(identifier):
    """Returns a algorithm.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A algorithm denoted by identifier.

    For example:
    >>> softlearning.algorithms.get({'class_name': 'SAC', ...})
      <softlearning.algorithms.sac.SAC object at 0x7fea93d6cdd0>
    >>> softlearning.algorithms.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown algorithm: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined algorithm.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(identifier)
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            f"Could not interpret algorithm function identifier:"
            " {repr(identifier)}.")
