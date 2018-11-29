from copy import deepcopy


def get_gaussian_policy(env, Q, observation_preprocessor=None, **kwargs):
    from .gaussian_policy import GaussianPolicy
    observation_shape = (
        env.active_observation_shape
        if observation_preprocessor is None
        else observation_preprocessor.output_shape[1:])
    policy = GaussianPolicy(
        input_shapes=(observation_shape, ),
        output_shape=env.action_space.shape,
        **kwargs)

    return policy


def get_uniform_policy(env, observation_preprocessor=None, **kwargs):
    from .uniform_policy import UniformPolicy
    observation_shape = (
        env.active_observation_shape
        if observation_preprocessor is None
        else observation_preprocessor.output_shape[1:])
    policy = UniformPolicy(
        input_shapes=(observation_shape, ),
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
                            *args,
                            **kwargs):
    policy_params = variant['policy_params']
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params['kwargs'])

    policy = POLICY_FUNCTIONS[policy_type](
        env,
        *args,
        Q=Qs[0],
        **policy_kwargs,
        **kwargs)

    return policy
