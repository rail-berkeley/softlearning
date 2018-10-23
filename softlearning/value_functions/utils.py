from . import vanilla


VALUE_FUNCTIONS = {
    'feedforward_value_function': (
        vanilla.create_feedforward_value_function),
    'double_feedforward_value_function': (
        vanilla.create_double_feedforward_value_function),
}


def get_value_function_from_params(params, input_shapes):
    kwargs = params.get('kwargs', {})
    value_function = VALUE_FUNCTIONS[params['type']](input_shapes, **kwargs)
    return value_function


def get_Q_function_from_variant(variant, env):
    Q_params = variant['Q_params'].copy()

    observation_shape = env.active_observation_shape
    action_shape = env.action_space.shape

    input_shapes = (observation_shape, action_shape)

    return get_value_function_from_params(Q_params, input_shapes)


def get_V_function_from_variant(variant, env):
    V_params = variant['V_params'].copy()

    observation_shape = env.active_observation_shape

    input_shapes = (observation_shape, )

    return get_value_function_from_params(V_params, input_shapes)
