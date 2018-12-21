import numpy as np
from gym.spaces import Box, Dict, Discrete, Tuple

from .flexible_replay_pool import FlexibleReplayPool


def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space

        observation_fields = normalize_observation_fields(observation_space)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool'
                },
            }
        }

        super(SimpleReplayPool, self).__init__(*args, fields=fields, **kwargs)

    def add_samples(self, num_samples, **kwargs):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(
                num_samples, **kwargs)

        kwargs.update(
           **{
               'observations.{}'.format(key): value
               for key, value in kwargs['observations'].items()
           },
           **{
               'next_observations.{}'.format(key): value
               for key, value in kwargs['next_observations'].items()
           },
        )

        del kwargs['observations']
        del kwargs['next_observations']

        return super(SimpleReplayPool, self).add_samples(
            num_samples, **kwargs)

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: getattr(self, field_name)[indices]
            for field_name in self.field_names
        }

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

        observations = np.concatenate([
            batch['observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        next_observations = np.concatenate([
            batch['next_observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        batch['observations'] = observations
        batch['next_observations'] = next_observations

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass
