from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)
from .adapters.rllab_adapter import (
    RLLAB_ENVIRONMENTS,
    RllabAdapter,
)


ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
    'rllab': RLLAB_ENVIRONMENTS
}

ADAPTERS = {
    'gym': GymAdapter,
    'rllab': RllabAdapter,
}


def get_environment(universe, domain, task, env_params, normalize=True):
    # TODO: Remember to normalize in the adapter
    return ADAPTERS[universe](domain, task, **env_params)
