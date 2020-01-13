import tensorflow as tf
import tree

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import create_inputs
from softlearning.utils.tensorflow import apply_preprocessors
from .base_value_function import StateActionValueFunction


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    inputs = create_inputs(input_shapes)
    if preprocessors is None:
        preprocessors = tree.map_structure(lambda _: None, inputs)

    preprocessed_inputs = apply_preprocessors(preprocessors, inputs)

    Q_model_body = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs
    )

    Q_model = tf.keras.Model(inputs, Q_model_body(preprocessed_inputs))

    Q_function = StateActionValueFunction(
        model=Q_model, observation_keys=observation_keys)

    return Q_function
