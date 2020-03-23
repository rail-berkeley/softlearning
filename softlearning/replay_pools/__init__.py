from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)

from .simple_replay_pool import SimpleReplayPool  # noqa: unused-import
from .goal_replay_pool import GoalReplayPool  # noqa: unused-import
from .union_pool import UnionPool  # noqa: unused-import
from .hindsight_experience_replay_pool import HindsightExperienceReplayPool  # noqa: unused-import


def serialize(replay_pool):
    return serialize_softlearning_object(replay_pool)


def deserialize(name, custom_objects=None):
    """Returns a replay pool function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Replay Pool function or class denoted by input string.

    For example:
    >>> softlearning.replay_pools.get({'class_name': 'SimpleReplayPool', ...})
      <softlearning.replay_pools.simple_replay_pool.SimpleReplayPool object at 0x7fea93d6cdd0>
    >>> softlearning.replay_pools.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown replay pool: abcd

    Args:
      name: The name of the replay pool.

    Raises:
        ValueError: `Unknown replay pool` if the input string does not
        denote any defined replay pool.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='replay pool')


def get(identifier):
    """Returns a replay pool.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A replay pool denoted by identifier.

    For example:
    >>> softlearning.replay_pools.get({'class_name': 'SimpleReplayPool', ...})
      <softlearning.replay_pools.simple_replay_pool.SimpleReplayPool object at 0x7fea93d6cdd0>
    >>> softlearning.replay_pools.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown replay_pool: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined replay pool.
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
            f"Could not interpret replay pool function identifier:"
            " {repr(identifier)}.")
