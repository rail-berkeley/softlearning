from softlearning.value_functions import NNQFunction, NNVFunction


def create_NNQFunction(variant, env):
    hidden_layer_sizes = variant['V_params']['hidden_layer_sizes']

    Qs = tuple(
        NNQFunction(
            observation_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf{}'.format(i))
        for i in range(2))

    return Qs


Q_FUNCTION_FUNCTIONS = {
    'NNQFunction': create_NNQFunction,
}


def get_Q_function_from_variant(variant, env):
    Q_params = variant['Q_params']
    Q_function = Q_FUNCTION_FUNCTIONS[Q_params['type']](variant, env)
    return Q_function


def create_NNVFunction(variant, env):
    hidden_layer_sizes = variant['V_params']['hidden_layer_sizes']

    V = NNVFunction(
        observation_shape=env.observation_space.shape,
        hidden_layer_sizes=hidden_layer_sizes)

    return V


V_FUNCTION_FUNCTIONS = {
    'NNVFunction': create_NNVFunction,
}


def get_V_function_from_variant(variant, env):
    V_params = variant['V_params']
    V_function = V_FUNCTION_FUNCTIONS[V_params['type']](variant, env)
    return V_function
