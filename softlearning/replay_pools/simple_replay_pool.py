from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool, Field


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        assert isinstance(observation_space, Dict), observation_space

        self._observation_space = observation_space
        self._action_space = action_space

        fields = {
            **{
                'observations': {
                    name: Field(
                        name=name,
                        dtype=observation_space.dtype,
                        shape=observation_space.shape)
                    for name, observation_space
                    in observation_space.spaces.items()
                },
                'next_observations': {
                    name: Field(
                        name=name,
                        dtype=observation_space.dtype,
                        shape=observation_space.shape)
                    for name, observation_space
                    in observation_space.spaces.items()
                },
                'actions': Field(
                    name='actions',
                    dtype=action_space.dtype,
                    shape=action_space.shape),
                'rewards': Field(
                    name='rewards',
                    dtype='float32',
                    shape=(1, )),
                # terminals[i] = a terminal was received at time i
                'terminals': Field(
                    name='terminals',
                    dtype='bool',
                    shape=(1, )),
            }
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)
