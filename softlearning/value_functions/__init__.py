from .vanilla import (  # noqa: unused-import
    feedforward_Q_function,
    double_feedforward_Q_function,
    ensemble_feedforward_Q_function,
)

from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)


def serialize(value_function):
    return serialize_softlearning_object(value_function)


def deserialize(name, custom_objects=None):
    """Returns a value function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Value function function or class denoted by input string.

    For example:
    >>> softlearning.value_functions.get('double_feedforward_Q_function')
      <function double_feedforward_Q_function at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Args:
      name: The name of the value function.

    Raises:
        ValueError: `Unknown value function` if the input string does not
        denote any defined value function.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='value function')


def get(identifier):
    """Returns a value function.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A value function denoted by identifier.

    For example:

    >>> softlearning.value_functions.get('double_feedforward_Q_function')
      <function double_feedforward_Q_function at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined value function.
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
            f"Could not interpret value function function identifier:"
            " {repr(identifier)}.")
