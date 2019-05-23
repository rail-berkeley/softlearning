from collections import OrderedDict
from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.models.utils import get_inputs_for_environment


def get_gaussian_policy(*args, **kwargs):
    from .gaussian_policy import FeedforwardGaussianPolicy

    policy = FeedforwardGaussianPolicy(*args, **kwargs)

    return policy


def get_uniform_policy(*args, **kwargs):
    from .uniform_policy import UniformPolicy

    policy = UniformPolicy(*args, **kwargs)

    return policy


POLICY_FUNCTIONS = {
    'GaussianPolicy': get_gaussian_policy,
    'UniformPolicy': get_uniform_policy,
}


def get_policy(policy_type, *args, **kwargs):
    return POLICY_FUNCTIONS[policy_type](*args, **kwargs)


def get_policy_from_params(policy_params, env, *args, **kwargs):
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params.get('kwargs', {}))

    observation_preprocessors_params = policy_kwargs.pop(
        'observation_preprocessors_params', {})
    observation_keys = policy_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    assert 'actions' not in observation_keys, observation_keys

    inputs, _ = get_inputs_for_environment(env)
    preprocessors = {key: None for key in inputs}

    for name, observation_input in inputs.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            continue
        preprocessor_params['kwargs']['inputs'] = observation_input
        preprocessor = get_preprocessor_from_params(env, preprocessor_params)
        preprocessors[name] = preprocessor

    if policy_type == 'UniformPolicy':
        action_range = (env.action_space.low, env.action_space.high)
        policy_kwargs['action_range'] = action_range

    inputs = [inputs[key] for key in sorted(inputs.keys())]
    preprocessors = [
        preprocessors[key] for key in sorted(preprocessors.keys())]

    policy = POLICY_FUNCTIONS[policy_type](
        inputs=inputs,
        output_shape=env.action_space.shape,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **policy_kwargs,
        **kwargs)

    return policy


def get_policy_from_variant(variant, *args, **kwargs):
    policy_params = variant['policy_params']
    return get_policy_from_params(policy_params, *args, **kwargs)
