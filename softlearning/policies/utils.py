from collections import OrderedDict
from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params


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

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue

        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    if policy_type == 'UniformPolicy':
        action_range = (env.action_space.low, env.action_space.high)
        policy_kwargs['action_range'] = action_range

    policy = POLICY_FUNCTIONS[policy_type](
        input_shapes=observation_shapes,
        output_shape=env.action_space.shape,
        observation_keys=observation_keys,
        *args,
        preprocessors=observation_preprocessors,
        **policy_kwargs,
        **kwargs)

    return policy


def get_policy_from_variant(variant, *args, **kwargs):
    policy_params = variant['policy_params']
    return get_policy_from_params(policy_params, *args, **kwargs)
