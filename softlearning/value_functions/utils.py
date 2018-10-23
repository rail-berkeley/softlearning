from copy import deepcopy

from . import vanilla


VALUE_FUNCTIONS = {
    'feedforward_value_function': (
        vanilla.create_feedforward_value_function),
    'double_feedforward_value_function': (
        vanilla.create_double_feedforward_value_function),
}


def get_Q_function_from_variant(variant, env, *args, **kwargs):
    Q_params = variant['Q_params']
    Q_type = Q_params['type']
    Q_kwargs = deepcopy(Q_params['kwargs'])

    input_shapes = (env.active_observation_shape, env.action_space.shape)

    return VALUE_FUNCTIONS[Q_type](input_shapes, **Q_kwargs, **kwargs)


def get_V_function_from_variant(variant, env, *args, **kwargs):
    V_params = variant['V_params']
    V_type = V_params['type']
    V_kwargs = deepcopy(V_params['kwargs'])

    input_shapes = (env.active_observation_shape, )

    return VALUE_FUNCTIONS[V_type](
        input_shapes, *args, **V_kwargs, **kwargs)
