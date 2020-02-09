from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)


def convnet_preprocessor(name='convnet_preprocessor', **kwargs):
    from softlearning.models.convnet import convnet_model

    preprocessor = convnet_model(name=name, **kwargs)

    return preprocessor


def serialize(preprocessor):
    return serialize_softlearning_object(preprocessor)


def deserialize(name, custom_objects=None):
    """Returns a preprocessor function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Preprocessor function or class denoted by input string.

    For example:
    >>> softlearning.preprocessors.get('convnet_preprocessor')
      <function convnet_preprocessor at 0x7fd170125950>
    >>> softlearning.preprocessors.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown preprocessor: abcd

    Args:
      name: The name of the preprocessor.

    Raises:
        ValueError: `Unknown preprocessor` if the input string does not
        denote any defined preprocessor.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='preprocessor')


def get(identifier):
    """Returns a preprocessor.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A preprocessor denoted by identifier.

    For example:

    >>> softlearning.preprocessors.get('convnet_preprocessor')
      <function convnet_preprocessor at 0x7fd170125950>
    >>> softlearning.preprocessors.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown preprocessor: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined preprocessor.
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
            f"Could not interpret preprocessor function identifier:"
            " {repr(identifier)}.")
