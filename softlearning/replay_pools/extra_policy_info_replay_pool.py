from serializable import Serializable
from .simple_replay_pool import SimpleReplayPool


class ExtraPolicyInfoReplayPool(SimpleReplayPool, Serializable):
    def __init__(self, *args, **kwargs):
        self._Serializable__initialize(locals())
        super(ExtraPolicyInfoReplayPool, self).__init__(*args, **kwargs)

        fields = {
            'raw_actions': {
                'shape': self._action_space.shape,
                'dtype': 'float32'
            },
            'log_pis': {
                'shape': (),
                'dtype': 'float32'
            }
        }

        self.add_fields(fields)
