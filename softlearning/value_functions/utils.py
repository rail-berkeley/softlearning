from collections import OrderedDict
from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params
from . import vanilla


def create_double_value_function(value_fn, *args, **kwargs):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(2))
    return value_fns


VALUE_FUNCTIONS = {
    'double_feedforward_Q_function': lambda *args, **kwargs: (
        create_double_value_function(
            vanilla.create_feedforward_Q_function, *args, **kwargs)),
}


def get_Q_function_from_variant(variant, env, *args, **kwargs):
    Q_params = deepcopy(variant['Q_params'])
    Q_type = deepcopy(Q_params['type'])
    Q_kwargs = deepcopy(Q_params['kwargs'])

    observation_preprocessors_params = Q_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = Q_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))
    action_shape = env.action_shape
    input_shapes = {
        'observations': observation_shapes,
        'actions': action_shape,
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    action_preprocessor = None
    preprocessors = {
        'observations': observation_preprocessors,
        'actions': action_preprocessor,
    }

    Q_function = VALUE_FUNCTIONS[Q_type](
        input_shapes=input_shapes,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **Q_kwargs,
        **kwargs)

    return Q_function
