from .convnet_preprocessor import ConvnetPreprocessor
from .mlp_preprocessor import (
    FeedforwardNetPreprocessor,
    FeedforwardNetPreprocessorV2,
)

PREPROCESSOR_FUNCTIONS = {
    'simple_convnet': ConvnetPreprocessor,
    'feedforward': FeedforwardNetPreprocessorV2,
}
