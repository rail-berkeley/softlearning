from softlearning.value_functions import NNQFunction, NNVFunction


def get_Q_function_from_variant(variant, env):
    layer_size = variant['value_fn_params']['layer_size']

    Qs = tuple(
        NNQFunction(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=(layer_size, layer_size),
            name='qf{}'.format(i))
        for i in range(2))

    return Qs


def get_V_function_from_variant(variant, env):
    layer_size = variant['value_fn_params']['layer_size']

    V = NNVFunction(
        observation_shape=env.observation_space.shape,
        hidden_layer_sizes=(layer_size, layer_size))

    return V
