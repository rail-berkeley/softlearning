from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

from .adapters.dm_control_adapter import (
    DM_CONTROL_ENVIRONMENTS,
    DmControlAdapter,
)


ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
    'dm_control': DM_CONTROL_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
    'dm_control': DmControlAdapter,
}


def get_environment(universe, domain, task, environment_params):
    return ADAPTERS[universe](domain, task, **environment_params)


def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, environment_kwargs)
