from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)

from .base_sampler import BaseSampler  # noqa: unused-import
from .dummy_sampler import DummySampler  # noqa: unused-import
from .simple_sampler import SimpleSampler  # noqa: unused-import
from .remote_sampler import RemoteSampler  # noqa: unused-import
from .utils import rollout, rollouts  # noqa: unused-import


def serialize(sampler):
    return serialize_softlearning_object(sampler)


def deserialize(name, custom_objects=None):
    """Returns a sampler function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Sampler function or class denoted by input string.

    For example:
    >>> softlearning.samplers.get({'class_name': 'SimpleSampler', ...})
      <softlearning.samplers.simple_sampler.SimpleSampler object at 0x7fea93d6cdd0>
    >>> softlearning.samplers.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown sampler: abcd

    Args:
      name: The name of the sampler.

    Raises:
        ValueError: `Unknown sampler` if the input string does not
        denote any defined sampler.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='sampler')


def get(identifier):
    """Returns a sampler.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A sampler denoted by identifier.

    For example:
    >>> softlearning.samplers.get({'class_name': 'SimpleSampler', ...})
      <softlearning.samplers.simple_sampler.SimpleSampler object at 0x7fea93d6cdd0>
    >>> softlearning.samplers.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown sampler: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined sampler.
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
            f"Could not interpret sampler function identifier:"
            " {repr(identifier)}.")
