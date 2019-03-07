from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}


def get_environment(universe, domain, task, environment_params):
    return ADAPTERS[universe](domain, task, **environment_params)


def get_environment_from_variant(variant):
    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']
    environment_params = variant['environment_params']

    return get_environment(universe, domain, task, environment_params)
