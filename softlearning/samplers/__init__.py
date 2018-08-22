from .sampler import Sampler
from .dummy_sampler import DummySampler
from .simple_sampler import SimpleSampler
from .remote_sampler import RemoteSampler
from .extra_policy_info_sampler import ExtraPolicyInfoSampler
from .utils import rollout, rollouts


SAMPLERS_BY_TYPE = {
    'Sampler': Sampler,
    'DummySampler': DummySampler,
    'SimpleSampler': SimpleSampler,
    'RemoteSampler': RemoteSampler,
    'ExtraPolicyInfoSampler': ExtraPolicyInfoSampler,
}


def get_sampler_from_params(sampler_params):
    sampler_type = sampler_params['type']
    sampler_args = sampler_params.get('args', ())
    sampler_kwargs = sampler_params.get('kwargs', {})
    sampler = SAMPLERS_BY_TYPE[sampler_type](
        *sampler_args, **sampler_kwargs)

    return sampler
