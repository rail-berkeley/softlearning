from .gaussian_policy import GaussianPolicy


def get_policy_from_variant(variant, env):
    policy_params = variant['policy_params']
    layer_size = variant['value_fn_params']['layer_size']

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(layer_size, layer_size),
            reparameterize=policy_params['reparameterize'])
    else:
        raise NotImplementedError(policy_params['type'])

    return policy
