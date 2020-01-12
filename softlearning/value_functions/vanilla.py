from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.keras import PicklableModel
from softlearning.utils.tensorflow import nest, apply_preprocessors


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    inputs = create_inputs(input_shapes)
    if preprocessors is None:
        preprocessors = nest.map_structure(lambda: None, inputs)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs, Q_function(preprocessed_inputs))
    Q_function.observation_keys = observation_keys

    return Q_function
