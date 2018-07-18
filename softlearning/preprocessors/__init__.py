from .mlp_preprocessor import (
    FeedforwardNetPreprocessor,
    FeedforwardNetPreprocessorV2,
)
from .convnet_preprocessor import convnet_preprocessor_template

PREPROCESSOR_FUNCTIONS = {
    'simple_convnet': convnet_preprocessor_template,
    'feedforward': FeedforwardNetPreprocessorV2,
}
