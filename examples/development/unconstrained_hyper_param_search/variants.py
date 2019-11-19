from ray import tune

from examples.development import (
    get_variant_spec as get_development_variant_spec)

def get_variant_spec(args):
    variant_spec = get_development_variant_spec(args)
    variant_spec['algorithm_params']['kwargs']['lr'] = tune.grid_search([1e-4, 3e-4, 5e-4])
    variant_spec['algorithm_params']['kwargs']['tau'] = tune.grid_search([1e-3, 5e-3, 1e-2])
    #variant_spec['algorithm_params']['sampler_params']['batch_size'] = tune.grid_search([128, 256, 512])
    # TODO check how to set seed
    return variant_spec