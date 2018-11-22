from copy import deepcopy

from .uniform_policy import UniformPolicy
from .gaussian_policy import GaussianPolicy


def get_gaussian_policy(env, Q, preprocessor, **kwargs):
    policy = GaussianPolicy(
        input_shapes=(env.active_observation_shape, ),
        output_shape=env.action_space.shape,
        **kwargs)

    return policy


def get_uniform_policy(env, *args, **kwargs):
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


def get_policy_from_variant(variant,
                            env,
                            Qs,
                            preprocessor,
                            *args,
                            **kwargs):
    policy_params = variant['policy_params']
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params['kwargs'])

    policy = POLICY_FUNCTIONS[policy_type](
        env,
        *args,
        Q=Qs[0],
        preprocessor=preprocessor,
        **policy_kwargs,
        **kwargs)

    return policy
