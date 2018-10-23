import tensorflow as tf


def feedforward_model(input_shapes,
                      hidden_layer_sizes,
                      output_size,
                      activation='relu',
                      output_activation='linear',
                      name=None,
                      *args,
                      **kwargs):
    if not isinstance(input_shapes[0], (list, tuple)):
        raise NotImplementedError(
            "TODO(hartikainen): feedforward_model currently expects a list of"
            " shapes as an input. It might be possible that you passed in a"
            " list/tuple of dimension objects. Those should be accepted"
            " but have not yet been implemented.")
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    if len(inputs) > 1:
        out = tf.keras.layers.Concatenate(axis=-1)(inputs)
    else:
        out = inputs[0]

    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, **kwargs)(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation, **kwargs)(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def feedforward_net_v2(inputs,
                       hidden_layer_sizes,
                       output_size,
                       activation='relu',
                       output_activation='linear',
                       *args,
                       **kwargs):
    if isinstance(inputs, (list, tuple)):
        inputs = tf.concat(inputs, axis=-1)

    out = inputs
    for units in hidden_layer_sizes:
        out = tf.layers.dense(
            inputs=out,
            units=units,
            activation=activation,
            *args,
            **kwargs)

    out = tf.layers.dense(
        inputs=out,
        units=output_size,
        activation=output_activation,
        *args,
        **kwargs)

    return out
