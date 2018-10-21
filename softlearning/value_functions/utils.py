import tensorflow as tf

from softlearning.misc.nn import feedforward_model
from . import metric_learning


def create_feedforward_Q_function(observation_shape,
                                  action_shape,
                                  name='Q',
                                  **kwargs):
    observations = tf.keras.layers.Input(
        shape=observation_shape, name='observations')
    actions = tf.keras.layers.Input(
        shape=action_shape, name='actions')

    Q_function = feedforward_model(
        [observations, actions], output_size=1, name=name, **kwargs)

    return Q_function


def create_double_feedforward_Q_function(*args, **kwargs):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    Qs = tuple(
        create_feedforward_Q_function(*args, name='Q{}'.format(i), **kwargs)
        for i in range(2))
    return Qs


def create_feedforward_V_function(observation_shape,
                                  name='V',
                                  **kwargs):
    observations = tf.keras.layers.Input(
        shape=observation_shape, name='observations')

    V_function = feedforward_model(
        [observations], output_size=1, name=name, **kwargs)

    return V_function


def create_feedforward_discriminator_function(observation_shape,
                                              num_skills,
                                              name='discriminator',
                                              **kwargs):
    observations = tf.keras.layers.Input(
        shape=observation_shape, name='observations')

    discriminator = feedforward_model(
        [observations], output_size=num_skills, **kwargs)

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
        Q_params['type']](
            env.active_observation_shape,
            env.action_space.shape,
            **kwargs)
    return Q_function


def get_V_function_from_variant(variant, env):
    V_params = variant['V_params'].copy()
    kwargs = V_params.get('kwargs', {})
    V_function = V_FUNCTION_FUNCTIONS[
        V_params['type']](
            env.active_observation_shape,
            **kwargs)
    return V_function
