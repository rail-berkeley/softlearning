import tensorflow as tf

from .mlp_preprocessor import feedforward_preprocessor_model


def get_convnet_preprocessor(observation_space,
                             name='convnet_preprocessor',
                             **kwargs):
    preprocessor = None
    raise NotImplementedError


def get_feedforward_preprocessor(observation_shape,
                                 name='feedforward_preprocessor',
                                 **kwargs):
    observations = tf.keras.layers.Input(
        shape=observation_shape, name='observations')

    preprocessor = feedforward_preprocessor_model(
        inputs=observations, name=name, **kwargs)

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'convnet_preprocessor': get_convnet_preprocessor,
    'feedforward_preprocessor': get_feedforward_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_variant(variant, env):
    preprocessor_params = variant['preprocessor_params']

    if not preprocessor_params:
        return None
    args = preprocessor_params.get('args', ())
    kwargs = preprocessor_params.get('kwargs', {})

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_params.get('type')](
            env.active_observation_shape, *args, **kwargs)

    return preprocessor
