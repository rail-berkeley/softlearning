import tensorflow as tf
import gym

from softlearning.misc.nn import feedforward_model
from . import metric_learning


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


def create_feedforward_Q_function(observation_space,
                                  action_space,
                                  name='Q',
                                  **kwargs):
    observations = observation_space_to_input(
        observation_space, name='observations')
    actions = action_space_to_input(action_space, name='actions')

    inputs = [observations, actions]

    Q_function = feedforward_model(inputs, output_size=1, name=name, **kwargs)

    return Q_function


def create_double_feedforward_Q_function(*args, **kwargs):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    Qs = tuple(
        create_feedforward_Q_function(*args, name='Q{}'.format(i), **kwargs)
        for i in range(2))
    return Qs


def create_feedforward_V_function(observation_space,
                                  name='V',
                                  **kwargs):
    observations = observation_space_to_input(
        observation_space, name='observations')

    inputs = [observations]

    V_function = feedforward_model(inputs, output_size=1, name=name, **kwargs)

    return V_function


def create_feedforward_discriminator_function(env,
                                              num_skills,
                                              name='discriminator',
                                              **kwargs):
    observations = observation_space_to_input(
        env.observation_space, name=name)

    inputs = [observations]

    discriminator = feedforward_model(
        inputs, output_size=num_skills, **kwargs)

    return discriminator


Q_FUNCTION_FUNCTIONS = {
    'feedforward_Q_function': create_feedforward_Q_function,
    'double_feedforward_Q_function': create_double_feedforward_Q_function,
    'metric_Q_function': metric_learning.create_metric_Q_function,
}


V_FUNCTION_FUNCTIONS = {
    'feedforward_V_function': create_feedforward_V_function,
    'metric_V_function': metric_learning.create_metric_V_function,
}


def get_Q_function_from_variant(variant, env):
    Q_params = variant['Q_params'].copy()
    kwargs = Q_params.get('kwargs', {})
    Q_function = Q_FUNCTION_FUNCTIONS[
        Q_params['type']](env.observation_space, env.action_space, **kwargs)
    return Q_function


def get_V_function_from_variant(variant, env):
    V_params = variant['V_params'].copy()
    kwargs = V_params.get('kwargs', {})
    V_function = V_FUNCTION_FUNCTIONS[
        V_params['type']](env.observation_space, **kwargs)
    return V_function
