from serializable import Serializable

from .flexible_replay_pool import FlexibleReplayPool


class SimpleReplayPool(FlexibleReplayPool, Serializable):
    def __init__(self, observation_shape, action_shape, *args, **kwargs):
        self._Serializable__initialize(locals())

        self._observation_shape = observation_shape
        self._action_shape = action_shape

        fields = {
            'observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            # It's a bit memory inefficient to save the observations twice,
            # but it makes the code *much* easier since you no longer have
            # to worry about termination conditions.
            'next_observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            'actions': {
                'shape': self._action_shape,
                'dtype': 'float32'
            },
            'rewards': {
                'shape': [],
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': [],
                'dtype': 'bool'
            },
        }

        super(SimpleReplayPool, self).__init__(*args, fields=fields, **kwargs)

    def terminate_episode(self):
        pass
