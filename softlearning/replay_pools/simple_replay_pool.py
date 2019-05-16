from collections import defaultdict
import copy

import numpy as np
from gym.spaces import Box, Dict, Discrete

from .flexible_replay_pool import FlexibleReplayPool, Field


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
            name: Field(
                name=name,
                dtype=observation_space.dtype,
                shape=observation_space.shape)
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
            'next_' + key: copy.deepcopy(value)
            for key, value in observation_fields.items()
        })
        observation_fields['next_observations'].name = 'next_observations'

        fields = {
            **observation_fields,
            **{
                'actions': Field(
                    name='actions',
                    dtype=action_space.dtype,
                    shape=action_space.shape),
                'rewards': Field(
                    name='rewards',
                    dtype='float32',
                    shape=(1, )),
                'terminals': Field(
                    name='terminals',
                    dtype='bool',
                    shape=(1, )),
            }
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields=fields, **kwargs)

    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        for observation in samples['observations']:
            for key, value in observation.items():
                dict_observations[key].append(value)

        dict_next_observations = defaultdict(list)
        for next_observation in samples['next_observations']:
            for key, value in next_observation.items():
                dict_next_observations[key].append(value)

        samples.update(
           **{
               f'observations.{observation_key}': np.array(values)
               for observation_key, values in dict_observations.items()
           },
           **{
               f'next_observations.{observation_key}': np.array(values)
               for observation_key, values in dict_next_observations.items()
           },
        )

        del samples['observations']
        del samples['next_observations']

        return super(SimpleReplayPool, self).add_samples(samples)

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
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
