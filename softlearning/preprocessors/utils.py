from copy import deepcopy

from softlearning.misc.nn import feedforward_model


def get_convnet_preprocessor(observation_space,
                             name='convnet_preprocessor',
                             **kwargs):
    preprocessor = None
    raise NotImplementedError


def get_feedforward_preprocessor(observation_shape,
                                 name='feedforward_preprocessor',
                                 **kwargs):
    preprocessor = feedforward_model(
        input_shapes=(observation_shape, ), name=name, **kwargs)

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'convnet_preprocessor': get_convnet_preprocessor,
    'feedforward_preprocessor': get_feedforward_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_variant(variant, env, *args, **kwargs):
    preprocessor_params = variant['preprocessor_params']
    preprocessor_type = preprocessor_params.get('type')
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if not preprocessor_params or preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            env.active_observation_shape,
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor
