from collections import OrderedDict
from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params
from softlearning.models.utils import get_inputs_for_environment
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
        'observation_preprocessors_params', {})
    observation_keys = Q_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    assert 'actions' not in observation_keys, observation_keys

    observation_inputs, action_input = get_inputs_for_environment(env)
    inputs = {**observation_inputs, 'actions': action_input}
    preprocessors = {key: None for key in inputs}

    for name, observation_input in observation_inputs.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            continue
        preprocessor_params['kwargs']['inputs'] = observation_input
        preprocessor = get_preprocessor_from_params(env, preprocessor_params)
        preprocessors[name] = preprocessor

    inputs = [inputs[key] for key in sorted(inputs.keys())]
    preprocessors = [
        preprocessors[key] for key in sorted(preprocessors.keys())]

    Q_function = VALUE_FUNCTIONS[Q_type](
        inputs=inputs,
        preprocessors=preprocessors,
        observation_keys=observation_keys,
        *args,
        **Q_kwargs,
        **kwargs)

    return Q_function
