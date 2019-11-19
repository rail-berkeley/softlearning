from ray import tune

from examples.development import (
    get_variant_spec as get_development_variant_spec)

def get_variant_spec(args):
    variant_spec = get_development_variant_spec(args)
    variant_spec['algorithm_params']['kwargs']['lr'] = tune.grid_search([1e-4, 3e-4, 5e-4])
    return variant_spec