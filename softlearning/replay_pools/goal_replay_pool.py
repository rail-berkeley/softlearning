from gym.spaces import Dict

from .flexible_replay_pool import FlexibleReplayPool, Field


class GoalReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 observation_fields=None,
                 new_observation_fields=None,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space
        assert isinstance(observation_space, Dict), observation_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

        fields = {
            'observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
                if name in environment.observation_keys
            },
            'next_observations': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
                if name in environment.observation_keys
            },
            'goals': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
                if name in environment.goal_keys
            },
            'actions': Field(
                name='actions',
                dtype=action_space.dtype,
                shape=environment.action_shape),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }

        super(GoalReplayPool, self).__init__(*args, fields=fields, **kwargs)

    def add_samples(self, samples, *args, **kwargs):
        observations = type(samples['observations'])(
            (key, values)
            for key, values in samples['observations'].items()
            if key in self._environment.observation_keys
        )
        next_observations = type(samples['next_observations'])(
            (key, values)
            for key, values in samples['next_observations'].items()
            if key in self._environment.observation_keys
        )
        goals = type(samples['observations'])(
            (key, values)
            for key, values in samples['observations'].items()
            if key in self._environment.goal_keys
        )

        samples.update({
            'observations': observations,
            'next_observations': next_observations,
            'goals': goals,
        })

        return super(GoalReplayPool, self).add_samples(
            samples, *args, **kwargs)
