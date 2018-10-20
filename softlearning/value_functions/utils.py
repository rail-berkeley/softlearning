import tensorflow as tf
import gym

from . import metric_learning


def create_value_function(inputs, hidden_layer_sizes, name=None):
    # concatenated = tf.keras.layers.Concatenate(axis=-1)(inputs)
    concatenated = (
        tf.keras.layers.Concatenate(axis=-1)(inputs)
        if len(inputs) > 1
        else inputs[0])

    out = concatenated
    for layer_size in hidden_layer_sizes:
        out = tf.keras.layers.Dense(layer_size, activation='relu')(out)

    out = tf.keras.layers.Dense(1, activation='linear')(out)

    model = tf.keras.Model(inputs, out, name=name)

    return model


def observation_space_to_input(observation_space, name='observations'):
    if isinstance(observation_space, gym.spaces.Dict):
        inputs = [
            tf.keras.layers.Input(
                shape=space.shape, name=f'{name}.{space_name}')
            for space_name, space in observation_space.spaces.items()
        ]
        input_ = tf.keras.layers.Concatenate(axis=-1)(inputs)
    else:
        input_ = tf.keras.layers.Input(
            shape=observation_space.shape, name=name)

    return input_


def action_space_to_input(action_space, name='actions'):
    input_ = tf.keras.layers.Input(shape=action_space.shape, name=name)
    return input_


def create_feedforward_Q_function(variant, env, name='Q'):
    observations = observation_space_to_input(
        env.observation_space, name='observations')
    actions = action_space_to_input(env.action_space, name='actions')

    inputs = [observations, actions]
    hidden_layer_sizes = variant['Q_params']['hidden_layer_sizes']

    Q_function = create_value_function(
        inputs, hidden_layer_sizes, name=name)

    return Q_function


def create_double_feedforward_Q_function(variant, env):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    Qs = tuple(
        create_feedforward_Q_function(variant, env, name='Q{}'.format(i))
        for i in range(2))
    return Qs


Q_FUNCTION_FUNCTIONS = {
    'feedforward_Q_function': create_feedforward_Q_function,
    'double_feedforward_Q_function': create_double_feedforward_Q_function,
    'metric_Q_function': metric_learning.create_metric_Q_function,
}


def get_Q_function_from_variant(variant, env):
    Q_params = variant['Q_params']
    Q_function = Q_FUNCTION_FUNCTIONS[Q_params['type']](variant, env)
    return Q_function


def create_feedforward_V_function(variant, env, name='V'):
    observations = observation_space_to_input(
        env.observation_space, name='observations')

    hidden_layer_sizes = variant['V_params']['hidden_layer_sizes']

    inputs = [observations]
    V_function = create_value_function(
        inputs, hidden_layer_sizes, name=name)

    return V_function


V_FUNCTION_FUNCTIONS = {
    'feedforward_V_function': create_feedforward_V_function,
    'metric_V_function': metric_learning.create_metric_V_function,
}


def get_V_function_from_variant(variant, env):
    V_params = variant['V_params']
    V_function = V_FUNCTION_FUNCTIONS[V_params['type']](variant, env)
    return V_function
