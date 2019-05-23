import tensorflow as tf


from softlearning.utils.keras import PicklableKerasModel


def feedforward_model(inputs,
                      output_size,
                      hidden_layer_sizes,
                      preprocessors=None,
                      activation='relu',
                      output_activation='linear',
                      name='feedforward_model',
                      *args,
                      **kwargs):
    if preprocessors is None:
        preprocessors = [None] * len(inputs)
    assert len(inputs) == len(preprocessors)
    preprocessed_inputs = [
        preprocessor(x) if preprocessor is not None else x
        for preprocessor, x in zip(preprocessors, inputs)
    ]
    concatenated = tf.keras.layers.Lambda(
        lambda inputs: tf.concat(
            [tf.cast(x, tf.float32) for x in inputs], axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model
