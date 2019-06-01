from gym.spaces import Dict

from .flexible_replay_pool import Field
from .simple_replay_pool import SimpleReplayPool


class GoalReplayPool(SimpleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 **kwargs):
        observation_space = environment.observation_space
        assert isinstance(observation_space, Dict), observation_space

        extra_fields = {
            'goals': {
                name: Field(
                    name=name,
                    dtype=observation_space.dtype,
                    shape=observation_space.shape)
                for name, observation_space
                in observation_space.spaces.items()
                if name in environment.goal_keys
            },
        }

        return super(GoalReplayPool, self).__init__(
            environment, *args, **kwargs, extra_fields=extra_fields)
