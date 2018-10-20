from .convnet_preprocessor import ConvnetPreprocessor
from .mlp_preprocessor import (
    FeedforwardNetPreprocessor,
    FeedforwardNetPreprocessorV2)


def get_simple_convnet_preprocessor(variant):
    preprocessor_params = variant['preprocessor_params']
    preprocessor_kwargs = preprocessor_params.get('kwargs', {})

    if 'num_conv_layers' in preprocessor_kwargs:
        num_conv_layers = preprocessor_kwargs.pop('num_conv_layers')
        filters_per_layer = preprocessor_kwargs.pop('filters_per_layer')
        kernel_size_per_layer = preprocessor_kwargs.pop('kernel_size_per_layer')

        conv_filters = (filters_per_layer, ) * num_conv_layers
        conv_kernel_sizes = (kernel_size_per_layer, ) * num_conv_layers
        preprocessor_kwargs['conv_filters'] = conv_filters
        preprocessor_kwargs['conv_kernel_sizes'] = conv_kernel_sizes

    preprocessor = ConvnetPreprocessor(
        *preprocessor_params.get('args', ()),
        **preprocessor_params.get('kwargs', {}))

    return preprocessor


def get_feedforward_preprocessor(variant):
    preprocessor_params = variant['preprocessor_params']

    preprocessor = FeedforwardNetPreprocessorV2(
        *preprocessor_params.get('args', ()),
        **preprocessor_params.get('kwargs', {}))

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'ConvnetPreprocessor': get_simple_convnet_preprocessor,
    'FeedforwardNetPreprocessorV2': get_feedforward_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_variant(variant):
    preprocessor_params = variant['preprocessor_params']

    if not preprocessor_params:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_params.get('type')](variant)

    return preprocessor
