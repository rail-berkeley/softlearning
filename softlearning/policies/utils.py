from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params


def get_gaussian_policy(env, Q, **kwargs):
    from .gaussian_policy import FeedforwardGaussianPolicy
    policy = FeedforwardGaussianPolicy(
        input_shapes=(env.active_observation_shape, ),
        output_shape=env.action_space.shape,
        **kwargs)

    return policy


def get_uniform_policy(env, *args, **kwargs):
    from .uniform_policy import UniformPolicy
    policy = UniformPolicy(
        input_shapes=(env.active_observation_shape, ),
        output_shape=env.action_space.shape)

    return policy


POLICY_FUNCTIONS = {
    'GaussianPolicy': get_gaussian_policy,
    'UniformPolicy': get_uniform_policy,
}


def get_policy(policy_type, *args, **kwargs):
    return POLICY_FUNCTIONS[policy_type](*args, **kwargs)


def get_policy_from_variant(variant, env, Qs, *args, **kwargs):
    policy_params = variant['policy_params']
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params['kwargs'])

    preprocessor_params = policy_kwargs.pop('preprocessor_params', None)
    preprocessor = get_preprocessor_from_params(env, preprocessor_params)

    policy = POLICY_FUNCTIONS[policy_type](
        env,
        *args,
        Q=Qs[0],
        preprocessor=preprocessor,
        **policy_kwargs,
        **kwargs)

    return policy
