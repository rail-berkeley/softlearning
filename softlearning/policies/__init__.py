from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)

from .base_policy import BasePolicy, LatentSpacePolicy, ContinuousPolicy  # noqa: unused-import
from .gaussian_policy import GaussianPolicy, FeedforwardGaussianPolicy  # noqa: unused-import
from .uniform_policy import UniformPolicyMixin, ContinuousUniformPolicy  # noqa: unused-import


def serialize(policy):
    return serialize_softlearning_object(policy)


def deserialize(name, custom_objects=None):
    """Returns a policy function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Policy function or class denoted by input string.

    For example:
    >>> softlearning.policies.get({
    ...     'class_name': 'ContinuousUniformPolicy',
    ...     'config': {
    ...         'action_range': [[-1], [1]],
    ...         'input_shapes': tf.TensorShape((3, )),
    ...         'output_shape': 2
    ...      }
    ... })
      <softlearning.policies.uniform_policy.ContinuousUniformPolicy object at 0x7fea93d6cdd0>
    >>> softlearning.policies.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown policy: abcd

    Args:
      name: The name of the policy.

    Raises:
        ValueError: `Unknown policy` if the input string does not
        denote any defined policy.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='policy')


def get(identifier):
    """Returns a policy.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A policy denoted by identifier.

    For example:
    >>> softlearning.policies.get({
    ...     'class_name': 'ContinuousUniformPolicy',
    ...     'config': {
    ...         'action_range': [[-1], [1]],
    ...         'input_shapes': tf.TensorShape((3, )),
    ...         'output_shape': 2
    ...      }
    ... })
      <softlearning.policies.uniform_policy.ContinuousUniformPolicy object at 0x7fea93d6cdd0>
    >>> softlearning.policies.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown policy: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined policy.
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
            f"Could not interpret policy function identifier:"
            " {repr(identifier)}.")
