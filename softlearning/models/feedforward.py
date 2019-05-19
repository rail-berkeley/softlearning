import tensorflow as tf


from softlearning.utils.keras import PicklableKerasModel
from softlearning.models.utils import flatten_input_structure


def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):

    input_shapes_flat = flatten_input_structure(input_shapes)
    inputs_flat = tuple(
        tf.keras.layers.Input(shape=shape) for shape in input_shapes_flat)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    model = PicklableKerasModel(inputs_flat, out, name=name)

    return model
